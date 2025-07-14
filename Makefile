install:
	poetry install

install-pre-commit:
	poetry run pre-commit install

lint:
	poetry run pre-commit run --all-files
