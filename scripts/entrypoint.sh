#!/bin/bash

set -e

# Wait for PostgreSQL to be available
echo "Waiting for PostgreSQL..."

until python -c "import socket; s=socket.socket(); s.connect(('${POSTGRES_HOST}', int(${POSTGRES_PORT})))" 2>/dev/null; do
  echo "Postgres is unavailable - sleeping"
  sleep 1
done

echo "Postgres is up - continuing..."

python manage.py makemigrations core
python manage.py makemigrations pipeline
python manage.py migrate
python manage.py create_superuser_if_debug
python manage.py runserver 0.0.0.0:8000
