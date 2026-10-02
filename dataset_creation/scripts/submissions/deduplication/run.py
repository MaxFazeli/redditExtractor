# Import your engine
from deduplication_engine import DeduplicationEngine

# Initialize it
engine = DeduplicationEngine(
    mongo_uri='mongodb://localhost:27017',
    db_name='reddit_data',
    input_collection='[14]image_url_removal',
    output_collection='[15]submissions_cleaned_full'
)

# Run the processing
engine.run()