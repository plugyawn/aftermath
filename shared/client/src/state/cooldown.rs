use crate::{HubUploadInfo, matformer::annotate_matformer_checkpoint_config};

use psyche_coordinator::{
    Coordinator,
    model::{self, HubRepo},
};
use psyche_core::{FixedString, NodeIdentity};
use psyche_data_provider::{UploadModelError, upload_model_repo_async};
use psyche_modeling::{
    SaveSafetensorsError, Trainer, TrainerThreadCommunicationError, save_tensors_into_safetensors,
};
use std::{
    cmp::Reverse,
    collections::{BinaryHeap, HashMap},
    path::PathBuf,
    sync::Arc,
};
use tch::Tensor;
use thiserror::Error;
use tokio::{
    sync::{Mutex, mpsc},
    task::JoinHandle,
};
use tracing::{Instrument, error, info, info_span, warn};

use super::{
    CheckpointConfig,
    evals::{ModelTaskRunner, RunningEvals},
};

#[derive(Error, Debug)]
pub enum CooldownError {
    #[error("no trainers available for checkpointing!")]
    NoTrainers,

    #[error("checkpointing thread crashed")]
    CheckpointThreadCrashed,

    #[error("error while checkpointing: {0}")]
    Checkpoint(#[from] CheckpointError),
}

/// MatFormer tier information for checkpoint saving.
#[derive(Clone, Copy, Debug)]
pub struct MatformerCheckpointInfo {
    /// The effective tier used during training (0 = full width).
    pub effective_tier: u8,
    /// The base intermediate size before any tier slicing.
    pub base_intermediate_size: Option<u64>,
    /// False when this client holds only the prefix of the FFN weights.
    pub holds_full_ffn_weights: bool,
}

impl MatformerCheckpointInfo {
    /// The coordinator keeps the last announced checkpoint for all later joins.
    ///
    /// A client with sliced weights has no FFN tail, so its upload would delete
    /// the tail from the run. It can save locally, but it must not publish.
    pub fn may_publish_checkpoint(&self) -> bool {
        self.holds_full_ffn_weights
    }
}

pub struct CooldownStepMetadata {
    tx_checkpoint: mpsc::UnboundedSender<model::HubRepo>,
    tx_model: mpsc::UnboundedSender<HashMap<String, Tensor>>,
    checkpoint_info: Option<CheckpointConfig>,
    checkpoint_extra_files: Vec<PathBuf>,
    matformer_info: MatformerCheckpointInfo,

    model_task_runner: ModelTaskRunner,
    // use a heap here as a best-effort attempt to ensure we get rid of the lowest step number dir even if we spawn multiple tasks
    // which may not finish writing their dirs in order. We note that even if we were to take the more complicated
    // route of actually enumerating the checkpoint_dir there would still be a race condition, unless we took a lockfile
    // or the like on the entire checkpoint_dir which probably isn't worth it just to support disk cleanup
    // we don't really expect there to be contention on this lock or real race conditions in practice though
    // as by the time one task spawns after a training round the previous write/upload task(s) should (hopefully) be long done
    delete_queue: Arc<Mutex<BinaryHeap<Reverse<u32>>>>,
}

impl CooldownStepMetadata {
    pub fn new(
        tx_checkpoint: mpsc::UnboundedSender<model::HubRepo>,
        tx_model: mpsc::UnboundedSender<HashMap<String, Tensor>>,
        checkpoint_info: Option<CheckpointConfig>,
        checkpoint_extra_files: Vec<PathBuf>,
        matformer_info: MatformerCheckpointInfo,
        model_task_runner: ModelTaskRunner,
    ) -> Self {
        Self {
            tx_checkpoint,
            tx_model,
            checkpoint_info,
            checkpoint_extra_files,
            matformer_info,
            model_task_runner,
            delete_queue: Arc::new(Mutex::new(BinaryHeap::new())),
        }
    }
}

#[derive(Error, Debug)]
pub enum CheckpointError {
    #[error("Extract thread crashed")]
    ExtractThreadCrashed,

    #[error("Trainer extract error: {0}")]
    Extract(#[from] TrainerThreadCommunicationError),

    #[error("Write thread crashed")]
    WriteThreadCrashed,

    #[error("Writing safetensors to disk failed: {0}")]
    WriteSafetensors(#[from] SaveSafetensorsError),

    #[error("Writing extra file to disk failed: {0}")]
    WriteExtraFile(tokio::io::Error),

    #[error("Reading config.json failed: {0}")]
    ReadConfigJson(tokio::io::Error),

    #[error("Parsing config.json failed: {0}")]
    ParseConfigJson(serde_json::Error),

    #[error("Writing config.json failed: {0}")]
    WriteConfigJson(tokio::io::Error),

    #[error("Couldn't upload model to huggingface: {0}")]
    UploadError(#[from] UploadModelError),

    #[error("Couldn't send checkpoint - channel closed")]
    SendCheckpoint,
}

async fn cleanup_dirs(
    delete_queue: Arc<Mutex<BinaryHeap<Reverse<u32>>>>,
    keep_steps: u32,
    run_id: String,
    delete_old_steps: bool,
    step: u32,
    checkpoint_dir: PathBuf,
) {
    if delete_old_steps {
        let mut delete_queue_guard = delete_queue.lock().await;
        delete_queue_guard.push(Reverse(step));
        // in the happy case this could be an if but if previous iterations failed somewhere
        // then we may have more than 1 dir to clean up
        while delete_queue_guard.len() > keep_steps as usize {
            let delete_step = delete_queue_guard.pop().unwrap().0;
            let delete_path = checkpoint_dir.join(format!("{run_id}-step{delete_step}"));
            if let Err(err) = tokio::fs::remove_dir_all(delete_path.clone()).await {
                warn!("Error removing {} : {}", delete_path.display(), err);
            } else {
                info!("Successfully removed {}", delete_path.display());
            }
        }
    }
}

impl CooldownStepMetadata {
    pub fn start<T: NodeIdentity>(
        &self,
        mut trainers: Vec<Trainer>,
        state: &Coordinator<T>,
    ) -> Result<CooldownStep, CooldownError> {
        let Some(mut trainer) = trainers.pop() else {
            return Err(CooldownError::NoTrainers);
        };

        let step = state.progress.step - 1;
        let run_id = String::from(&state.run_id);
        let checkpoint_extra_files = self.checkpoint_extra_files.clone();
        let checkpoint_info = self.checkpoint_info.clone();
        let matformer_info = self.matformer_info;
        let tx_checkpoint = self.tx_checkpoint.clone();
        let tx_model = self.tx_model.clone();
        let model_task_runner = self.model_task_runner.clone();
        let delete_queue = self.delete_queue.clone();

        let checkpointing_and_evals: CheckpointAndEvalsHandle = tokio::task::spawn(
            async move {
                info!("Extracting full model...");
                let (variables, trainer) =
                    tokio::task::spawn_blocking::<_, Result<_, CheckpointError>>(|| {
                        let variables = trainer.extract()?;
                        info!("Model extracted; {} parameters", variables.len());
                        Ok((variables, trainer))
                    })
                    .await
                    .map_err(|_| CheckpointError::ExtractThreadCrashed)??;

                let variables_clone: HashMap<String, Tensor> = variables
                    .iter()
                    .map(|(name, var)| (name.clone(), var.shallow_clone()))
                    .collect();

                trainers.push(trainer);
                let evals = model_task_runner.start(trainers);

                tx_model
                    .send(variables_clone)
                    .map_err(|_| CheckpointError::SendCheckpoint)?;

                let Some(CheckpointConfig {
                    hub_upload,
                    checkpoint_dir,
                    delete_old_steps,
                    keep_steps,
                }) = checkpoint_info
                else {
                    // If there was no HF checkpointing configuration, return immediately
                    return Ok((evals, None));
                };

                // Start the upload process of the updated model parameters in a separate task
                let upload_handle = tokio::task::spawn(async move {
                    let path = checkpoint_dir.join(format!("{run_id}-step{step}"));
                    info!("Saving to {}", path.display());
                    let mut local = tokio::task::spawn_blocking({
                        let path = path.clone();
                        move || save_tensors_into_safetensors(variables, path)
                    })
                    .await
                    .map_err(|_| CheckpointError::WriteThreadCrashed)??;

                    for extra in checkpoint_extra_files {
                        let filename = extra.file_name().unwrap();
                        let to = path.join(filename);

                        // Handle config.json specially: inject matformer_tier info
                        if filename == "config.json" {
                            let content = tokio::fs::read_to_string(&extra)
                                .await
                                .map_err(CheckpointError::ReadConfigJson)?;
                            let mut config: serde_json::Value = serde_json::from_str(&content)
                                .map_err(CheckpointError::ParseConfigJson)?;

                            annotate_matformer_checkpoint_config(
                                &mut config,
                                matformer_info.base_intermediate_size,
                            );

                            let updated = serde_json::to_string_pretty(&config)
                                .map_err(CheckpointError::ParseConfigJson)?;
                            tokio::fs::write(&to, updated)
                                .await
                                .map_err(CheckpointError::WriteConfigJson)?;
                        } else {
                            tokio::fs::copy(extra.clone(), to.clone())
                                .await
                                .map_err(CheckpointError::WriteExtraFile)?;
                        }
                        local.push(to);
                    }

                    let hub_upload = if matformer_info.may_publish_checkpoint() {
                        hub_upload
                    } else {
                        if hub_upload.is_some() {
                            warn!(
                                effective_tier = matformer_info.effective_tier,
                                "This client holds sliced MatFormer weights, so it will not upload this checkpoint"
                            );
                        }
                        None
                    };
                    let Some(HubUploadInfo {
                        hub_repo,
                        hub_token,
                    }) = hub_upload
                    else {
                        cleanup_dirs(
                            delete_queue,
                            keep_steps,
                            run_id,
                            delete_old_steps,
                            step,
                            checkpoint_dir,
                        )
                        .await;
                        return Ok::<(), CheckpointError>(());
                    };

                    info!(repo = hub_repo, "Uploading checkpoint to HuggingFace");
                    let revision = match upload_model_repo_async(
                        hub_repo.clone(),
                        local,
                        hub_token.clone(),
                        Some(format!("step {step}")),
                        None,
                    )
                    .await
                    {
                        Ok(revision) => {
                            info!(
                                repo = hub_repo,
                                revision = revision,
                                "Upload to HuggingFace complete"
                            );
                            revision
                        }
                        Err(err) => {
                            error!(repo = hub_repo, "Error uploading to HuggingFace: {err:#}");
                            return Err(err.into());
                        }
                    };

                    tx_checkpoint
                        .send(HubRepo {
                            repo_id: FixedString::from_str_truncated(&hub_repo),
                            revision: Some(FixedString::from_str_truncated(&revision)),
                        })
                        .map_err(|_| CheckpointError::SendCheckpoint)?;

                    // we put the cleanup step at the end, so that if keep_steps == 0 the logic will still work
                    // we'll just delete the dir after we've uploaded it
                    // if we fail in any of the above steps we may wind up not queueing this dir for delete
                    // but that's probably better than risking having the dir deleted from under us
                    // for a relatively low priority disk cleanup task
                    // and this may actually be preferred anyway because if we failed to upload, we may want to keep
                    // the data around locally on disk
                    cleanup_dirs(
                        delete_queue,
                        keep_steps,
                        run_id,
                        delete_old_steps,
                        step,
                        checkpoint_dir,
                    )
                    .await;

                    Ok(())
                });

                Ok((evals, Some(upload_handle)))
            }
            .instrument(info_span!("checkpointing")),
        );
        Ok(CooldownStep {
            checkpointing_and_evals,
        })
    }
}

type CheckpointAndEvalsHandle = JoinHandle<
    Result<
        (
            RunningEvals,
            Option<JoinHandle<Result<(), CheckpointError>>>,
        ),
        CheckpointError,
    >,
>;

#[derive(Debug)]
pub struct CooldownStep {
    checkpointing_and_evals: CheckpointAndEvalsHandle,
}

impl CooldownStep {
    pub async fn finish(
        self,
    ) -> Result<
        (
            RunningEvals,
            Option<JoinHandle<Result<(), CheckpointError>>>,
        ),
        CooldownError,
    > {
        let (running_evals, upload_handle) = self
            .checkpointing_and_evals
            .await
            .map_err(|_| CooldownError::CheckpointThreadCrashed)??;

        Ok((running_evals, upload_handle))
    }
}

#[cfg(test)]
mod tests {
    use super::MatformerCheckpointInfo;

    fn info(holds_full_ffn_weights: bool) -> MatformerCheckpointInfo {
        MatformerCheckpointInfo {
            effective_tier: 0,
            base_intermediate_size: Some(1024),
            holds_full_ffn_weights,
        }
    }

    #[test]
    fn only_a_client_with_full_weights_can_publish_a_checkpoint() {
        assert!(info(true).may_publish_checkpoint());
        assert!(!info(false).may_publish_checkpoint());
    }
}
