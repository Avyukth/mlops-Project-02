#!/bin/bash
set -e

echo 'MLflow configuration:'
echo "MLFLOW_TRACKING_URI=$MLFLOW_TRACKING_URI"
echo "MLFLOW_DEFAULT_ARTIFACT_ROOT=$MLFLOW_DEFAULT_ARTIFACT_ROOT"
echo "MLFLOW_S3_ENDPOINT_URL=$MLFLOW_S3_ENDPOINT_URL"

# Ensure the database directory exists and is writable
mkdir -p "$(dirname "$MLFLOW_TRACKING_URI" | sed 's|sqlite:///||')"
chmod 777 "$(dirname "$MLFLOW_TRACKING_URI" | sed 's|sqlite:///||')"

exec mlflow server \
	--backend-store-uri "$MLFLOW_TRACKING_URI" \
	--default-artifact-root "$MLFLOW_DEFAULT_ARTIFACT_ROOT" \
	--host 0.0.0.0 \
	--port 6060
