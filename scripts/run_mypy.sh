#!/bin/bash
export DJANGO_SETTINGS_MODULE=backend.settings
export PYTHONPATH="$PYTHONPATH:$(pwd)"

venv/bin/python3 -m mypy backend core