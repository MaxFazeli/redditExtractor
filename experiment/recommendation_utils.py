"""
Utility functions for the recommendation experiment.
"""

import csv
import os
from datetime import datetime

# Columns of a result CSV. Fixed, so that every row (including error rows) has the same
# structure and rows can be appended one at a time.
RESULT_FIELDNAMES = [
    'submission_id',
    'domain',
    'model',
    'prompt_type',
    'input_mode',
    'submission_text',
    'few_shot_example_ids',
    'valid_json',
    'valid_structure',
    'nr_recs',
    'nr_recs_unique',
    'nr_ground_truth',
    'llm_recommendations',
    'ground_truth',
    'raw_response',
    'finish_reason',
    'response_truncated',
    'prompt_truncated',
    'prompt_tokens',
    'response_tokens',
    'duration_seconds',
    'error',
    'timestamp',
]


class ResultWriter:
    """Appends result rows to a CSV file, one row at a time.

    Every row is flushed and fsync'ed immediately, so a crash, a SLURM time-out or a container
    stop loses at most the submission currently being processed. On restart, rows that are already
    in the file are reported via `done_ids` so that they can be skipped (resume).
    """

    def __init__(self, filepath):
        self.filepath = filepath
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        self.done_ids = set()

        if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
            with open(filepath, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                if reader.fieldnames != RESULT_FIELDNAMES:
                    # File written by an older version of the script: keep it, but start a new one.
                    backup = f"{filepath}.{datetime.now().strftime('%Y%m%d_%H%M%S')}.old_format"
                    f.close()
                    os.replace(filepath, backup)
                    print(f"Existing result file has a different column layout; moved it to {backup}")
                else:
                    self.done_ids = {row['submission_id'] for row in reader}

        new_file = not os.path.exists(filepath) or os.path.getsize(filepath) == 0
        self._file = open(filepath, 'a', newline='', encoding='utf-8')
        self._writer = csv.DictWriter(self._file, fieldnames=RESULT_FIELDNAMES, restval='')
        if new_file:
            self._writer.writeheader()
            self._sync()

    def write(self, row):
        unknown = set(row) - set(RESULT_FIELDNAMES)
        if unknown:
            raise ValueError(f"Unknown result columns: {sorted(unknown)}")
        self._writer.writerow(row)
        self._sync()
        self.done_ids.add(str(row.get('submission_id')))

    def _sync(self):
        self._file.flush()
        os.fsync(self._file.fileno())

    def close(self):
        if not self._file.closed:
            self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def create_directory_if_not_exists(directory):
    """Create directory if it doesn't exist."""
    os.makedirs(directory, exist_ok=True)
