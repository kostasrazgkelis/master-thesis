#!/bin/bash

set -e

# Wait for PostgreSQL to be available
echo "Waiting for PostgreSQL at $POSTGRES_HOST:$POSTGRES_PORT..."

until poetry run python -c "import socket; s=socket.socket(); s.settimeout(3); s.connect(('${POSTGRES_HOST}', int(${POSTGRES_PORT})))" 2>/dev/null; do
  echo "Postgres is unavailable - sleeping"
  sleep 1
done

echo "Postgres is up - continuing..."

poetry run python manage.py makemigrations core
poetry run python manage.py makemigrations pipeline
poetry run python manage.py migrate
poetry run python manage.py create_superuser_if_debug
poetry run python manage.py runserver 0.0.0.0:8000
