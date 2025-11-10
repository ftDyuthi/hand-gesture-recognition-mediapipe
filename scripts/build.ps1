# Ensure dependencies are installed (do this if you haven't already)
poetry install --no-root

# Run the module with `--` so poetry doesn't interpret `-m`
poetry run -- python -m tools.poetry_build -- --mode onedir