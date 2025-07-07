FROM python:3.13.5-slim

WORKDIR /app

# Install system dependencies for PostgreSQL
RUN apt-get update && apt-get install -y \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Change to backend directory where manage.py is located
WORKDIR /app/backend

# Expose Django port
EXPOSE 8000

# Run Django development server
CMD ["sh", "-c", "python manage.py migrate && python manage.py create_superuser_if_debug && python manage.py runserver 0.0.0.0:8000"]
