# TextOps

## Description

TextOps is a production simulation project designed to bridge the gap between model experimentation and reliable ML systems. While the core task is text classification, the primary focus is on engineering practices such as reproducibility, automation and lifecycle management.

Quick problem summary: a skewed classification problem with 4 classes, 2 of which are significantly underrepresented, with a low total sample size (~800).

## Feature Highlights

- **Gradio**-based web application for clean UI, straightforward user interaction
<br> <br>
- **FastAPI** handles front end requests and provides live predictions on user input
<br> <br>
- **MLFlow**
    - _experiment tracking_ : full reproducibility and detailed history
    - _model registry_ : clean overview of produced models, each with associated run info, parameters, artifacts, metrics
    - _automatic promotion_ and serving when training produces a model that surpasses the current champion
    - _S3 artifact storage_ : using **SeaweedFS**, MLFlow artifacts are stored in and retrieved from the _mlflow-artifacts_ S3 bucket
<br> <br>
- **DVC**
    - _pipeline_ : clean, separated training stages. Small alterations don't require the whole training to rerun, only affected stages.
    - _data versioning_
    - _artifact tracking_
    - _CLI visualization_, comparison of metrics, plots of past runs
<br> <br>
- **Docker Compose** orchestration
    - _separation of duties_ : front end, API, MLFlow server, training
    - the whole setup, simple, clean, reproducible, on any machine
<br> <br>
- **VS Code Dev Containers** : unified development, identical environment with simple setup
<br> <br>
- **GitHub Actions** : automatically runs tests on pull requests  _coming soon_
<br> <br>
- **pre-commit** : automatic code checking and formatting, using Ruff and yamllint
<br> <br>
- **uv** : dependency management
- **pytest** : quality assurance
<br> <br>
- **Ruff** : Python linting and code formatting
- **PyDantic** : inference type checking
- **Pylance** : in strict mode, for consistent typing
- **yamllint** : YAML linting
<br> <br>
- Cached text embedding generation during training


## Introduction

### Getting Started

To set up the core application, simply run:

```bash
docker compose up
```

This boots up the app, API, SeaweedFS and MLFlow services.


### Connecting

When running locally, you can then connect to the application using a browser, located at [http://localhost:7860/](http://localhost:7860/)
When running over SSH, you can connect via `http://<your_server_ip>:7860/`.
You can get your IP address on Linux by running `hostname -I`. Then replace '<your_server_ip>' with the first displayed IP.

To checkout the MLFlow server, showcasing training and evaluation results, produced models, the current champion as well as logged graphs and metrics, visit [http://localhost:7860/](http://localhost:7860/), or with your respective IP address.


At first, the MLFlow server will look a tad barren. To produce some training results, you'll have to run training.


### Training

For this, there's two options. Either via smart staging, only executing altered stages, using:

```bash
docker compose \
    --profile train-pipe \
    run --rm train-pipe
```

Or, you can run the entire pipeline regardless of changes, using:

```bash
docker compose \
    --profile train-pipe \
    run --rm train-pipe -f
```

(the exact same command, just with a `-f` at the end, or `--force` if you will)


### Customizing

Training runs a search by default. Parameters are defined in the `params.yaml` file. Feel free to experiment, altering search parameters, adding more etc.

I've included a champion model in the repository _coming soon_. With some tweaks, you can certainly produce a model with better performance.


### Automatic Promotion

At the end of the training pipeline, the model gets evaluated and pitted against the current champion. If the new model surpasses the past champion, it will automatically be promoted: it gets highlighted in MLFlow, will be loaded by FastAPI on future runs, and can be tested live in the app; all without any code changes.


## Plots

While the project is still WIP; here's some nice plots to look at, automatically generated after training and logged through MLFlow when applicable:

![Confusion Matrix](plots/confusion_matrix.png)

![Precision-Recall Curve](plots/precision_recall_curve.png)

![Roc Curve](plots/roc_curve.png)

![Tree-Based Feature Importance](plots/tree_based_feature_importance.png)
