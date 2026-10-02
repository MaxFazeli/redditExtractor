from pymongo import MongoClient
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed


def classify_submission_by_frequency(submission):
    # Combine only string values from the submission
    combined_text = ' '.join(str(value).lower() for value in submission.values() if isinstance(value, str))

    # Keyword lists for all categories
    category_keywords = {
        "Health and Fitness": ["fitness", "exercise", "workout", "gym", "yoga", "nutrition", "wellness",
                               "diet", "meditation", "Nike", "Adidas", "Under Armour", "Fitbit", "Peloton"],
        "Business and Finance": ["business", "finance", "investment", "stocks", "market", "economy", "entrepreneur",
                                 "crypto", "money"],
        "Art": ["painting", "sculpture", "gallery", "drawing", "design", "museum"],
        "Automobiles": ["vehicle", "automobile", "engine", "motor", "electric vehicle",
                        "Toyota", "Ford", "Honda", "BMW", "Mercedes", "Tesla", "Chevrolet"],
        "Music": [
            "music", "remix", "melody", "dance", "lyric", "lyrics", "musical", "band", "song","songs", "album", "artist", "concert", "voice",
            "rap", "rapper", "rock", "pop", "jazz", "blues", "classical",
            "hip hop", "country", "electronic", "reggae", "metal", "indie",
            "punk", "folk", "R&B", "soul", "gospel", "singing","soundtrack", "spotify", "vocal", "vocals"
        ],
        "Environment": ["climate", "sustainability", "recycling"],
        "Movies": [
            "movie", "movies", "film", "cinema", "director", "actor", "imdb",
            "netflix", "action", "comedy", "drama", "horror", "thriller",
            "romance", "sci-fi", "fantasy", "animation", "documentary",
            "mystery", "adventure", "historical", "biography"
        ],
        "Books": [
            "book", "books", "novel", "author", "literature", "fiction", "non-fiction",
            "mystery", "thriller", "fantasy", "science fiction", "biography",
            "romance", "historical", "young adult", "self-help", "poetry",
            "graphic novel", "children's"
        ],
        "Games": [
            "game", "games", "video game", "playstation", "xbox", "gaming",
            "steam", "pc game", "role-playing game", "strategy", "action",
            "adventure", "simulation", "sports", "multiplayer",
            "single-player", "indie", "MMORPG", "open-world"
        ],
        "Sports": ["sport", "sports", "soccer", "football", "basketball", "tennis", "olympics"],
        "Fashion": ["fashion", "clothing", "designer", "Gucci", "Chanel", "Nike", "Adidas", "Prada"],
        "Travel": ["travel", "destination", "vacation", "holiday", "trip"],
        "TV Shows": ["tv show", "drama", "drams", "show", "shows", "series", "episode", "season", "comedian", "anime"],
    }

    # Count occurrences of keywords in each category and keep track of matched keywords
    category_counts = defaultdict(int)
    matched_keywords = {}

    # Create regex pattern to match all keywords at once
    for category, keywords in category_keywords.items():
        pattern = r'\b(?:' + '|'.join(map(re.escape, keywords)) + r')\b'
        found_keywords = re.findall(pattern, combined_text)

        if found_keywords:
            category_counts[category] += len(found_keywords)
            matched_keywords[category] = list(set(found_keywords))  # Store unique matched keywords

    # Ensure category_counts is not empty before finding the max
    if category_counts:
        max_category = max(category_counts, key=category_counts.get)
        return (max_category if category_counts[max_category] > 0 else "Other", matched_keywords)
    else:
        return ("Other", {})  # Return "Other" if no keywords matched


def process_submission(submission):
    classified_submission = submission.copy()  # Copy the existing submission
    domain_area, matched_keywords = classify_submission_by_frequency(submission)  # Classify submission
    classified_submission['domain_area'] = domain_area
    classified_submission['matched_keywords'] = matched_keywords  # Add matched keywords
    classified_submission.pop('_id', None)  # Remove the _id field to avoid duplicates
    return classified_submission


# Connect to MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['reddit_data']  # Replace with your database name
submissions_collection = db['submissions_no_null']  # Replace with your existing collection name
classified_collection = db['classified_submissions']  # New collection for classified submissions

# Fetch all submissions in batches to avoid memory issues
batch_size = 1000  # Adjust the batch size based on your data size
cursor = submissions_collection.find()
batch = []

for submission in cursor:
    batch.append(submission)
    if len(batch) == batch_size:
        # Use ThreadPoolExecutor to process submissions in parallel
        with ThreadPoolExecutor(max_workers=10) as executor:  # Adjust max_workers as needed
            future_to_submission = {executor.submit(process_submission, sub): sub for sub in batch}

            for future in as_completed(future_to_submission):
                classified_submission = future.result()
                # Insert the new document into the classified collection
                classified_collection.insert_one(classified_submission)

        batch.clear()  # Clear the batch for the next set of submissions

# Process remaining submissions in the last batch
if batch:
    with ThreadPoolExecutor(max_workers=10) as executor:  # Adjust max_workers as needed
        future_to_submission = {executor.submit(process_submission, sub): sub for sub in batch}

        for future in as_completed(future_to_submission):
            classified_submission = future.result()
            classified_collection.insert_one(classified_submission)

print("Classified submissions have been stored in the new collection using multithreading.")
