# Analysis Folder

This folder contains all analysis scripts and statistical results for the Reddit entity classification project.

## Folder Structure

### `scripts/`
Contains Python analysis scripts for data extraction and analysis:

- **`analyze_gpt_responses.py`** - Analyzes GPT response patterns to understand classification logic
- **`check_batch_inclusion.py`** - Checks if comments were included in batch processing files
- **`classification_stats.py`** - Main analysis script with complete classification statistics
- **`extract_classification_numbers.py`** - Extracts basic classification numbers from MongoDB
- **`find_comment_examples.py`** - Finds examples of different comment classification states
- **`classification_missing_types.py`** - Detailed analysis distinguishing between different types of missing classifications

### `statistics/`
Contains analysis results and statistical reports:

- **`classification_stats.json`** - Complete classification data in JSON format

## Key Findings

### Classification Success Rates:
- **Submissions**: 94.6% (1,765/1,865)
- **Comments**: 81.6% (25,131/30,795)

### Domain Performance (Comment Classification):
1. **Books**: 61.0% (most challenging)
2. **Music**: 71.1%
3. **Film**: 85.0%
4. **Games**: 87.2%
5. **TV**: 87.9% (best performance)

### Missing Comment Types:
- **No Field**: 3,652 comments (11.9%) - GPT processed but returned null/empty
- **Empty Field**: 2,012 comments (6.5%) - GPT processed but found no entities
- **Total Missing**: 5,664 comments (18.4%)

## Usage

To run the main analysis:
```bash
cd analysis/scripts
python classification_stats.py
```

This will generate the complete classification statistics and save results to `../statistics/classification_stats.json`.
