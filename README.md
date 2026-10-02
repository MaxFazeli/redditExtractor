# redditExtractor

Master thesis, TU Graz: recommendations from r/ifyoulikeblank as ground truth for evaluating
LLM-based recommendations.

## Structure

```
dataset_creation/   pipeline that built the dataset and ground truth (filtering steps, GPT labelling batches)
analysis/           analysis scripts and statistics of the dataset creation
experiment/         LLM recommendation experiment (code, prompts, Docker/SLURM setup)
docs/               experiment-pipeline.md (how the experiment runs), changelog.md (changes and decisions)
archive/            old / superseded files, see archive/README.md
data/               raw data, MongoDB dumps, dataset exports (local only, not in git)
```

## Dataset

MongoDB database `reddit_data`, collection `dataset`: 1,718 submissions
(Books 92, Film 450, Games 269, Music 437, TV 470), each with `title`, `selftext`,
`link_flair_text` (domain) and `final_recommendations` (ground truth).

## Experiment

20 open-weight models via Ollama on nvcluster and 3 OpenAI models (list in
`experiment/framework/models.py`), zero-shot and few-shot prompts per domain.

- How it works: [docs/experiment-pipeline.md](docs/experiment-pipeline.md)
- How to run it: [experiment/setup/docker/README.md](experiment/setup/docker/README.md)
- Changes and decisions: [docs/changelog.md](docs/changelog.md)

```bash
cd experiment
python -m unittest tests/test_units.py -v
python recommendation_evaluator.py --results-dir setup/results/full_run
```

## Subreddit extraction (raw data)

* Info about .zstd files from [https://files.pushshift.io/reddit/submissions/README.txt]()
* **UPDATE 17.08.2023**: pushshift not available anymore! Use torrents instead (search for "reddit torrents 2005 2023).


**WARNING: 24 threads are too much --> memory error --> fixed: readline() instead of readlines()**

**WARNING: 24 threads are too much --> HDD out of space --> use 12 threads --> set variable in extract_subreddit.py**


1. Docker with Ubuntu:

	`docker run --rm -it --name container_name -v /path/to/Reddit-Extractor_python3:/path/to/Reddit-Extractor_python3 -v /media/nas_datasets/datasets/Reddit:/media/nas_datasets/datasets/Reddit ubuntu:22.04`

2. Install zstd (v1.4.1)

	`apt update`
	
	`apt install -y zstd`

	`cd /path/to/Reddit-Extractor_python3/`


3. Install Python (3.8)

	`apt install -y python3 python3-pip`
	
	Install python packages with pip

4. Run scripts:

	Edit both scripts first: add subreddits to extract (variable "subreddits")

	`time python3 extract_subreddit.py`

	`time python3 prettify_output.py`
