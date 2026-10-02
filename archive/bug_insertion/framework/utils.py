#!/usr/bin/env python
# encoding: utf-8

import os
import shutil
import pandas as pd

from logger import debug_msg, logging


def getDurationString(start_time, end_time):
    duration = end_time - start_time
    days = duration.days
    hours, remainder = divmod(duration.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    duration_string = ""
    duration_string += " %sd" % days if days > 0 else ""
    duration_string += " %sh" % hours if hours > 0 else ""
    duration_string += " %sm" % minutes if minutes > 0 else ""
    duration_string += " %ss" % seconds if seconds > 0 or duration_string.strip() == "" else ""
    return duration_string.strip()


def deleteExistingDirectory(directory, create_new=False):
    if os.path.exists(directory):
        debug_msg("Directory %s already exists. Delete..." % directory, level=logging.DEBUG)
        shutil.rmtree(directory)
    if create_new:
        debug_msg("Create (empty) directory: %s" % directory, level=logging.DEBUG)
        os.mkdir(directory)


def create_directory_if_not_exists(directory_path):
    """Create the specified directory if it does not exist."""
    try:
        os.makedirs(directory_path, exist_ok=True)
        return True
    except Exception as e:
        print(f"An error occurred: {e}")
        return False
    

def read_csv(file_path):
    """Function to read CSV file."""
    return pd.read_csv(file_path)


def save_csv(df, file_path, index=False):
    """Save the DataFrame to a CSV file."""
    df.to_csv(file_path, index=index)


def save_json(df, file_path, orient='records', lines=True):
    """Save the DataFrame to a CSV file."""
    df.to_json(file_path, orient=orient, lines=lines)
