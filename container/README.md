# Docker Container Setup

This directory contains Docker configuration files for running the Entity Matching Pipeline in a containerized environment.

## Files

- `docker-compose.yml`: Docker Compose configuration for easy container orchestration

## Quick Start

### Using Docker Compose

1. **Navigate to the container directory**:
```bash
cd container
```

2. **Start the container**:
```bash
docker-compose up
```

3. **Access Jupyter Lab**:
   - Open your browser to `http://localhost:8888`
   - The project files will be available in the `/home/jovyan/work` directory inside the container

4. **Access Spark UI**:
   - Spark Web UI is available at `http://localhost:4040` when Spark is running

## Container Specifications

- **Base Image**: `jupyter/pyspark-notebook:latest`
- **Java Version**: OpenJDK 11
- **Spark Version**: 3.3.2
- **Memory Limit**: 4GB
- **CPU Limit**: 3 cores
- **Additional**: GraphFrames library included

## Port Mappings

- `8888`: Jupyter Lab interface
- `4040`: Spark Web UI

## Volume Mounting

The parent directory (containing the project files) is mounted to `/home/jovyan/work` inside the container, allowing you to:
- Access all project files and notebooks
- Save changes persistently
- Run experiments directly from the container

## Stopping the Container

```bash
# If using docker-compose
docker-compose down

# If using docker run, press Ctrl+C or use:
docker stop <container_id>
```

## Troubleshooting

- **Memory Issues**: Increase the `mem_limit` in docker-compose.yml if needed
- **Port Conflicts**: Change the port mappings if 8888 or 4040 are already in use
- **Permission Issues**: Ensure Docker has permission to mount the project directory
