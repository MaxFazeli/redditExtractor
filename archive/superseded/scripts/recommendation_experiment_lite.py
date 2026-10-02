#!/usr/bin/env python
# encoding: utf-8

"""
Lightweight Recommendation Experiment - Quick Test Version
Only runs: 10 random movie submissions with zero-shot prompts
"""

import sys
import os
import random
import json
import time
from pymongo import MongoClient
from datetime import datetime

# Import from framework
# Add framework to path using absolute path
script_dir = os.path.dirname(os.path.abspath(__file__))
framework_path = os.path.join(script_dir, 'framework')
sys.path.insert(0, framework_path)

from ollama_client import OllamaClient
from openai_client import OpenAiClient

# MongoDB connection
MONGO_HOST = 'mongodb'  # Docker service name
MONGO_PORT = 27017
MONGO_DB = 'reddit_data'
MONGO_COLLECTION = 'dataset'

# Prompt file (absolute path using script_dir from above)
MOVIE_PROMPT_FILE = os.path.join(script_dir, 'prompts/zero_shot_prompts/movie_tv_zero_shot_test.txt')

# Output directory (file name will be dynamic with timestamp and model)
RESULTS_DIR = os.path.join(script_dir, 'setup/results')


def load_prompt():
    """Load the movie zero-shot prompt template."""
    if not os.path.exists(MOVIE_PROMPT_FILE):
        raise FileNotFoundError(f"Prompt file not found: {MOVIE_PROMPT_FILE}")
    
    with open(MOVIE_PROMPT_FILE, 'r', encoding='utf-8') as f:
        return f.read()


def format_prompt(template, submission):
    """Format prompt template with submission data."""
    submission_text = f"Title: {submission.get('title', '')}\n"
    
    if submission.get('selftext'):
        submission_text += f"Description: {submission['selftext']}\n"
    
    return template.replace('{submission_text}', submission_text.strip())


def generate_recommendations(prompt, model_name, ollama_client, openai_client):
    """Call LLM and get recommendations."""
    # Determine which client to use
    if model_name.startswith('gpt') or model_name.startswith('GPT'):
        response = openai_client.chat(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
    else:
        response = ollama_client.generate(
            model=model_name,
            system_prompt="You are a helpful recommendation assistant. Always respond with valid JSON.",
            user_prompt=prompt
        )
    
    return response


def parse_recommendations(llm_response):
    """Parse LLM response and extract recommendations list."""
    if not llm_response.ok:
        return {
            'success': False,
            'error': llm_response.finish_reason or 'Unknown error',
            'recommendations': []
        }
    
    try:
        response_text = llm_response.message.strip()
        
        # Handle markdown code blocks
        if response_text.startswith('```'):
            lines = response_text.split('\n')
            response_text = '\n'.join(lines[1:-1]) if len(lines) > 2 else response_text
        
        # Parse JSON
        parsed = json.loads(response_text)
        
        # Extract recommendations array
        if isinstance(parsed, list):
            recommendations = parsed
        elif isinstance(parsed, dict):
            recommendations = parsed.get('recommendations', 
                            parsed.get('items', 
                            parsed.get('results', [])))
        else:
            recommendations = []
        
        return {
            'success': True,
            'recommendations': recommendations,
            'raw_response': response_text
        }
    
    except json.JSONDecodeError as e:
        return {
            'success': False,
            'error': f'JSON parse error: {str(e)}',
            'recommendations': [],
            'raw_response': llm_response.message
        }


def wait_for_ollama(ollama_client, max_retries=30, delay=5):
    """Wait for Ollama to be ready before proceeding."""
    print("[1.5/6] Waiting for Ollama to be ready...")
    for attempt in range(1, max_retries + 1):
        try:
            # Try to list models as a health check
            ollama_client.client.list()
            print(f"  ✅ Ollama is ready (attempt {attempt})")
            return True
        except Exception as e:
            if attempt < max_retries:
                print(f"  ⏳ Ollama not ready yet (attempt {attempt}/{max_retries}), retrying in {delay}s...")
                time.sleep(delay)
            else:
                print(f"  ❌ Ollama failed to become ready after {max_retries} attempts")
                print(f"     Error: {e}")
                return False
    return False


def main():
    """Main experiment workflow."""
    print("=" * 80)
    print("LIGHTWEIGHT RECOMMENDATION EXPERIMENT")
    print("=" * 80)
    print(f"Domain: Film/TV")
    print(f"Submissions: 10 random")
    print(f"Prompt type: Zero-shot")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 80)
    print()
    
    # Initialize clients
    ollama_client = OllamaClient()
    openai_client = OpenAiClient()
    
    # Wait for Ollama to be ready
    if not wait_for_ollama(ollama_client):
        print("  ❌ Cannot proceed without Ollama. Exiting.")
        return
    print()
    
    # Connect to MongoDB
    print("[2/6] Connecting to MongoDB...")
    try:
        client = MongoClient(MONGO_HOST, MONGO_PORT, serverSelectionTimeoutMS=5000)
        client.server_info()
        db = client[MONGO_DB]
        collection = db[MONGO_COLLECTION]
        print(f"  ✅ Connected to {MONGO_DB}.{MONGO_COLLECTION}")
    except Exception as e:
        print(f"  ❌ MongoDB connection failed: {e}")
        return
    
    # Load prompt template
    print("[3/6] Loading zero-shot prompt...")
    try:
        prompt_template = load_prompt()
        print(f"  ✅ Loaded from {MOVIE_PROMPT_FILE}")
    except Exception as e:
        print(f"  ❌ Failed to load prompt: {e}")
        return
    
    # Get 10 random movie/TV submissions
    print("[4/6] Fetching 10 random Film/TV submissions...")
    try:
        # Query for Film or TV submissions
        movie_submissions = list(collection.find({
            'link_flair_text': {'$in': ['Film', 'TV']}
        }))
        
        if len(movie_submissions) < 10:
            print(f"  ⚠️  Only {len(movie_submissions)} Film/TV submissions found")
            random_submissions = movie_submissions
        else:
            random_submissions = random.sample(movie_submissions, 10)
        
        print(f"  ✅ Selected {len(random_submissions)} submissions")
        for i, sub in enumerate(random_submissions, 1):
            title = sub.get('title', 'No title')[:60]
            print(f"     {i}. {title}...")
    except Exception as e:
        print(f"  ❌ Failed to fetch submissions: {e}")
        return
    
    # Use hardcoded tiny model
    print()
    print("[5/6] Model selection")
    model_name = "llama3.3:70b-instruct-q4_0"
    print(f"  ✅ Using model: {model_name}")    
    
    # Create output filename with timestamp and model name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_safe_name = model_name.replace(":", "_").replace("/", "_")
    output_filename = f"lite_test_{model_safe_name}_{timestamp}.json"
    output_file = os.path.join(RESULTS_DIR, output_filename)
    print(f"  📝 Output file: {output_filename}")
    
    # Load/pull the model if needed
    print(f"  📥 Loading model {model_name}...")
    if ollama_client.load_model(model_name):
        print(f"  ✅ Model ready")
    else:
        print(f"  ❌ Failed to load model {model_name}")
        return
    print()
    
    # Process submissions
    print(f"[6/6] Running experiment with {model_name}...")
    results = []
    
    for i, submission in enumerate(random_submissions, 1):
        print(f"  [{i}/{len(random_submissions)}] Processing submission {submission['_id']}...")
        
        # Format prompt
        prompt = format_prompt(prompt_template, submission)
        
        # Get recommendations
        try:
            llm_response = generate_recommendations(prompt, model_name, ollama_client, openai_client)
            parsed = parse_recommendations(llm_response)
            
            # Store result
            result = {
                'submission_id': str(submission['_id']),
                'title': submission.get('title', ''),
                'domain': submission.get('link_flair_text', ''),
                'model': model_name,
                'prompt_type': 'zero_shot',
                'success': parsed['success'],
                'recommendations': parsed['recommendations'],
                'error': parsed.get('error'),
                'raw_response': parsed.get('raw_response', '')[:500]  # Truncate
            }
            
            results.append(result)
            
            if parsed['success']:
                print(f"      ✅ Got {len(parsed['recommendations'])} recommendations")
            else:
                print(f"      ❌ Failed: {parsed.get('error', 'Unknown')}")
        
        except Exception as e:
            print(f"      ❌ Exception: {str(e)}")
            results.append({
                'submission_id': str(submission['_id']),
                'title': submission.get('title', ''),
                'domain': submission.get('link_flair_text', ''),
                'model': model_name,
                'success': False,
                'error': str(e)
            })
    
    # Save results
    print()
    print("=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)
    
    # Create results directory if needed
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    output_data = {
        'timestamp': datetime.now().isoformat(),
        'model': model_name,
        'domain': 'Film/TV',
        'prompt_type': 'zero_shot',
        'total_submissions': len(random_submissions),
        'successful': sum(1 for r in results if r['success']),
        'failed': sum(1 for r in results if not r['success']),
        'results': results
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Results saved to: {output_file}")
    print()
    print("Summary:")
    print(f"  Total: {len(results)}")
    print(f"  Successful: {output_data['successful']}")
    print(f"  Failed: {output_data['failed']}")
    print("=" * 80)


if __name__ == '__main__':
    main()
