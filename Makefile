.PHONY: install
install:
	poetry install

.PHONY: install-pre-commit
install-pre-commit:
	poetry run pre-commit uninstall; poetry run pre-commit install

.PHONY: lint
lint:
	poetry run pre-commit run --all-files

.PHONY: migrate
migrate:
	poetry run python -m backend.manage migrate

.PHONY: migrations
migrations:
	poetry run python -m backend.manage makemigrations

.PHONY: run-server
run-server:
	poetry run python -m backend.manage runserver

.PHONY: shell
shell:
	poetry run python -m backend.manage shell

.PHONY: test
test:
	poetry run pytest -v -rs -n auto --show-capture=no

.PHONY: up-dependencies-only
up-dependencies-only:
	test -f .env || touch .env
	docker-compose -f docker-compose.yml up --force-recreate db

.PHONY: update
update: install migrate install-pre-commit ;