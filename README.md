# Colorizer

Colorizer is an education project aimed at implementing convolutional neural network that
will colorize grayscale images with high accuracy. This project is being created as the main assignment of
"Missing ML Semester" cource of HSE AMI bachelor program.

# Knowledge base
Build.in pages:
- [View](https://buildin.ai/dfbakin/share/87cb8381-036a-4ab2-b72e-25eee01672dc?code=KHTJB8)
- [Edit](https://buildin.ai/87cb8381-036a-4ab2-b72e-25eee01672dc) (for collaborators only)

# How to download datasets?
- Install DVC with pipx
```bash
pipx install dvc[s3]
```
- And then download all all data or specific .dvc files
```bash
dvc install # only once from the repository's root

dvc pull # to download all files
dvc pull <filename>.dvc # to download specific model or dataset
```

# How to run?
Install environment (we use poetry)
- https://python-poetry.org/docs/#installing-with-pipx
- https://pipx.pypa.io/stable/installation/

Then use `poetry` to run any python script:
```bash
poetry run ...
```

# How to collaborate?
Install [pre-commit](https://pre-commit.com/) and run `pre-commit install`.
Now on any commit your files will be formatted properly.

Then push you changes to a branch, open pull request.
