sync:
    uv run jupytext --sync notebooks/exploration.ipynb

fix:
    uv run ruff check . --fix
    uv run ruff format .
