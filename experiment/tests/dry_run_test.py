#!/usr/bin/env python
# encoding: utf-8

"""
Dry run test - validates setup without calling LLMs.
Checks MongoDB connection, prompts, data structure.
"""

from pymongo import MongoClient
import os
import json
import sys

# Change to experiment directory so relative paths work
script_dir = os.path.dirname(os.path.abspath(__file__))
experiment_dir = os.path.dirname(script_dir)
os.chdir(experiment_dir)

print("=" * 80)
print("DRY RUN - Recommendation Experiment Validation")
print("=" * 80)
print()

# Test 1: MongoDB Connection
print("[1/6] Testing MongoDB connection...")
try:
    client = MongoClient(os.environ.get('MONGODB_HOST', 'localhost'), int(os.environ.get('MONGODB_PORT', '27017')), serverSelectionTimeoutMS=5000)
    client.server_info()
    print("  [OK] MongoDB is accessible")
except Exception as e:
    print(f"  [FAIL] MongoDB connection failed: {e}")
    print("  → Start MongoDB, e.g.: docker compose -f setup/docker/docker-compose.mongodb-test.yml up -d")
    exit(1)

# Test 2: Dataset Collection
print("[2/6] Checking dataset collection...")
try:
    db = client['reddit_data']
    dataset = db['dataset']
    count = dataset.count_documents({})
    print(f"  [OK] Found {count} submissions in dataset")
    
    if count == 0:
        print("  [FAIL] Dataset is empty!")
        exit(1)
except Exception as e:
    print(f"  [FAIL] Dataset check failed: {e}")
    exit(1)

# Test 3: Sample Submission Structure
print("[3/6] Validating submission structure...")
try:
    sample = dataset.find_one()
    required_fields = ['id', 'title', 'link_flair_text', 'final_recommendations']
    missing = [f for f in required_fields if f not in sample]
    
    if missing:
        print(f"  [FAIL] Missing required fields: {missing}")
        exit(1)
    
    print(f"  [OK] Sample submission:")
    print(f"     ID: {sample['id']}")
    print(f"     Domain: {sample['link_flair_text']}")
    print(f"     Title: {sample['title'][:60]}...")
    print(f"     Ground truth recs: {len(sample['final_recommendations'])}")
except Exception as e:
    print(f"  [FAIL] Structure validation failed: {e}")
    exit(1)

# Test 4: Domain Distribution
print("[4/6] Checking domain distribution...")
try:
    domains = dataset.distinct('link_flair_text')
    print(f"  [OK] Found domains: {domains}")
    
    for domain in domains:
        count = dataset.count_documents({'link_flair_text': domain})
        print(f"     {domain}: {count} submissions")
except Exception as e:
    print(f"  [FAIL] Domain check failed: {e}")
    exit(1)

# Test 5: Prompt Files
print("[5/6] Checking prompt files...")
prompt_checks = {
    'Books zero-shot': 'prompts/zero_shot_prompts/books_zero_shot.txt',
    'Books few-shot': 'prompts/few_shot_prompts/books_few_shot.txt',
    'Film/TV zero-shot': 'prompts/zero_shot_prompts/movie_tv_zero_shot.txt',
    'Film/TV few-shot': 'prompts/few_shot_prompts/movie_tv_few_shot.txt',
    'Music zero-shot': 'prompts/zero_shot_prompts/music_zero_shot.txt',
    'Music few-shot': 'prompts/few_shot_prompts/music_few_shot.txt',
    'Games zero-shot': 'prompts/zero_shot_prompts/games_zero_shot.txt',
    'Games few-shot': 'prompts/few_shot_prompts/games_few_shot.txt',
}

all_prompts_ok = True
for name, path in prompt_checks.items():
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
            if '<submission_text>' in content:
                print(f"  [OK] {name}: OK ({len(content)} chars)")
            else:
                print(f"  [WARN]  {name}: Missing <submission_text> placeholder")
                all_prompts_ok = False
    else:
        print(f"  [FAIL] {name}: File not found")
        all_prompts_ok = False

if not all_prompts_ok:
    print("  → Some prompts are missing or malformed")
    exit(1)

# Test 6: Few-Shot Example Selection
print("[6/6] Testing few-shot example selection...")
try:
    domain = 'Books'
    sample_id = sample['id']
    
    # Get 3 examples
    cursor = dataset.find({
        'link_flair_text': domain,
        'id': {'$ne': sample_id}
    }).limit(3)
    
    examples = list(cursor)
    print(f"  [OK] Found {len(examples)} example submissions for domain '{domain}'")
    
    if len(examples) < 3:
        print(f"  [WARN]  Warning: Only {len(examples)} examples available (need 3 for few-shot)")
    
    # Show first example
    if examples:
        ex = examples[0]
        print(f"     Example: {ex['title'][:50]}...")
        print(f"     Example recs: {len(ex.get('final_recommendations', []))}")
except Exception as e:
    print(f"  [FAIL] Example selection failed: {e}")
    exit(1)

print()
print("=" * 80)
print("[OK] ALL CHECKS PASSED!")
print("=" * 80)
print()
print("Your setup is ready. Next steps:")
print()
print("1. Test with real LLM (requires Ollama or OpenAI):")
print("   python tests/test_recommendation_experiment.py")
print()
print("2. Or run manually:")
print("   python recommendation_experiment.py --limit 5 --model llama3.2:3b-instruct-q4_K_M --output-dir test_results")
print()
print("3. Or with OpenAI:")
print("   export OPENAI_API_KEY='your-key'")
print("   python recommendation_experiment.py --limit 5 --model openai:gpt-5-nano --output-dir test_results")
print()
