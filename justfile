sync:
    uv sync --all-packages

sync-notebook:
    uv run jupytext --sync notebooks/exploration.ipynb

fix:
    uv run ruff check . --fix
    uv run ruff format .

commit:
    just fix
    git add .
    git status

train:
    uv run dvc repro

run_mlflow:
    (cd ./mlruns && \
    uv run mlflow server \
        --host 0.0.0.0 \
        --port "5000" \
        --backend-store-uri sqlite:///./mlflow.db \
        --artifacts-destination ./mlartifacts)

run_api:
    uv run --package api uvicorn api.api:app

run_app:
    uv run --package app python -m app.app

run_app_w_hot_reload:
    uv run --package app gradio app/src/app/app.py
