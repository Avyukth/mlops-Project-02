#!/bin/bash

# Sleep for 5 minutes
# sleep 200

# Start the API server
uvicorn api.main:app --host 0.0.0.0 --port 8000
