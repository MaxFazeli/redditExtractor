# Dataset Creation Statistics

## Overview
This document tracks the statistics for the final dataset creation process, where submissions from `[15]submissions_cleaned_full` were filtered based on the number of final recommendations.

## Date
January 26, 2026

## Filtering Criteria
- **Source Collection**: `[15]submissions_cleaned_full`
- **Target Collection**: `dataset`
- **Filter Condition**: Submissions with `final_recommendations` array containing >= 5 entries

## Results

### Summary Statistics
- **Total Submissions Processed**: 1,865
- **Submissions Included**: 1,718 (92.1%)
- **Submissions Excluded**: 147 (7.9%)

### Detailed Breakdown
- **Included in Dataset**: 1,718 submissions with 5 or more recommendations
- **Kicked Due to Condition**: 147 submissions with less than 5 recommendations

## Script
- **Location**: `scripts/submissions/sub_15_filter_final_recommendations.py`
- **Input**: `[15]submissions_cleaned_full` collection
- **Output**: `dataset` collection
