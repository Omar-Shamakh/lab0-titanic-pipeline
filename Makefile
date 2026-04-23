format:
	uv run isort src/ main.py
	uv run black src/ main.py
	uv run ruff check --fix src/ main.py

train:
	uv run python main.py

lint:
	uv run ruff check src/ main.py