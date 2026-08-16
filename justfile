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

host:
    mlflow server \
     --host 0.0.0.0 \
     --port 5000 \
     --default-artifact-root ./mlruns \
     --backend-store-uri sqlite:///./mlruns/mlflow.db

run_api:
    uv run --package api api

run_app:
    uv run --package app app

run_app_w_hot_reload:
    uv run --package app gradio app/src/app/app.py --demo-name=app
