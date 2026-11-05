# Building the Psyche Book

That's the document you're reading! :D

## Development

Install the tools once:

```bash
just book_deps
```

Then serve the book over http on localhost:

```bash
source scripts/psyche-env.sh
just serve_book
```

This recipe builds the CLIs to generate the command-line reference, so you must have PyTorch installed.

## Building

`nix build .#psyche-book`

The book will be output to `result/`, which you can preview easily with `python -m http.server -d ./result/`

## Publishing

The `Docs` GitHub Actions workflow deploys the book to <https://plugyawn.github.io/aftermath/> after each push to `main` that changes the book. You can also start the workflow manually from the Actions tab.
