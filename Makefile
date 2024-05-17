lint:
	poetry run mypy .
	poetry run ruff check .
	poetry run isort --check .

format:
	poetry run isort .
	poetry run ruff format .
	poetry run ruff check --fix .