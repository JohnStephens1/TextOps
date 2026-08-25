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
- **uv** : dependency management
- **Ruff** : linting and code formatting
<br> <br>
- **PyDantic** : inference type checking
- **pytest** : quality assurance
<br> <br>
- Cached text embedding generation during training

## Plots

While the project is still WIP, since the README's looking a tad barren; here's some nice plots to look at, automatically generated after training and logged through MLFlow when applicable:

![Confusion Matrix](plots/confusion_matrix.png)

![Precision-Recall Curve](plots/precision_recall_curve.png)

![Roc Curve](plots/roc_curve.png)

![Tree-Based Feature Importance](plots/tree_based_feature_importance.png)
