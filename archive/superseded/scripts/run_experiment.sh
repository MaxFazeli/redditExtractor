#!/bin/bash
# Run the recommendation experiment
# This script is called after Ollama port forwarding is established

echo "=================================="
echo "Starting Recommendation Experiment"
echo "=================================="

# Wait a bit for Ollama to be ready
sleep 5

# Run the experiment with all models
cd /app/experiment
python recommendation_experiment.py

echo "=================================="
echo "Experiment Complete!"
echo "Results saved to: /app/results"
echo "=================================="
