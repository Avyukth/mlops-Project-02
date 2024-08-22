#!/bin/bash
set -e

# Upgrade the MLflow database schema
mlflow db upgrade $MLFLOW_TRACKING_URI

# Run the main application
exec python main.py
