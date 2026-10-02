#!/usr/bin/env python
# encoding: utf-8

"""
Test script for recommendation_experiment.py
Runs a small-scale test with 1 model and 5 submissions.
"""

import subprocess
import sys
import os

# Change to experiment directory so relative paths work
script_dir = os.path.dirname(os.path.abspath(__file__))
experiment_dir = os.path.dirname(script_dir)
os.chdir(experiment_dir)

def run_test():
    """Run a small test of the recommendation experiment."""
    print("=" * 80)
    print("RECOMMENDATION EXPERIMENT - TEST RUN")
    print("=" * 80)
    print()
    print("Configuration:")
    print("  - Model: llama3.2:3b-instruct-q4_K_M (single model test)")
    print("  - Submissions: 5 (limited for quick test)")
    print("  - Prompt types: zero_shot, few_shot")
    print("  - Output: ./test_results/")
    print()
    print("This will:")
    print("  1. Connect to MongoDB localhost:27017")
    print("  2. Load 5 submissions from 'dataset' collection")
    print("  3. Generate recommendations using llama3.2:3b-instruct-q4_K_M")
    print("  4. Save results to CSV files")
    print()
    
    response = input("Continue? (y/n): ")
    if response.lower() != 'y':
        print("Test cancelled.")
        return
    
    print()
    print("Starting test...")
    print("-" * 80)
    
    # Run the experiment
    cmd = [
        sys.executable,
        'recommendation_experiment.py',
        '--model', 'llama3.2:3b-instruct-q4_K_M',
        '--prompt-types', 'zero_shot', 'few_shot',
        '--limit', '5',
        '--output-dir', 'test_results'
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print()
        print("=" * 80)
        print("TEST COMPLETE!")
        print("=" * 80)
        print()
        print("Results saved to: ./test_results/")
        print()
        print("Check the following files:")
        print("  - test_results/llama3.2_3b-instruct-q4_K_M/zero_shot_results.csv")
        print("  - test_results/llama3.2_3b-instruct-q4_K_M/few_shot_results.csv")
        print()
        print("Next steps:")
        print("  1. Inspect the CSV files")
        print("  2. Check if recommendations look reasonable")
        print("  3. Verify JSON parsing worked correctly")
        print("  4. If everything looks good, run full experiment!")
        print()
        
    except subprocess.CalledProcessError as e:
        print()
        print("=" * 80)
        print("TEST FAILED!")
        print("=" * 80)
        print()
        print(f"Error: {e}")
        print()
        print("Common issues:")
        print("  - MongoDB not running: see setup/docker/docker-compose.mongodb-test.yml")
        print("  - Ollama not accessible: Check container/port forwarding")
        print("  - Missing dependencies: pip install -r framework/requirements.txt")
        print()
        sys.exit(1)


if __name__ == '__main__':
    run_test()
