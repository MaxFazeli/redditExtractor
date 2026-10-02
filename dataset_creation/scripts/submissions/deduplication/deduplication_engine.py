from pymongo import MongoClient
from rapidfuzz import fuzz
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class DeduplicationEngine:
    def __init__(self, mongo_uri, db_name, input_collection, output_collection):
        self.client = MongoClient(mongo_uri)
        self.db = self.client[db_name]
        self.input_collection = self.db[input_collection]
        self.output_collection = self.db[output_collection]

        self.category_map = {
            'Film': {
                'id_field': 'IMDb',
                'title_field': 'title',
                'year_field': 'year',
                'creator_field': 'director'
            },
            'TV': {
                'id_field': 'IMDb',
                'title_field': 'title',
                'year_field': 'year',
                'creator_field': 'director'
            },
            'Books': {
                'id_field': 'ISBN',
                'title_field': 'title',
                'year_field': 'year',
                'creator_field': 'author'
            },
            'Music': {
                'id_field': 'ID',
                'title_field': 'entity',
                'year_field': 'year',
                'creator_field': 'entity'
            },
            'Games': {
                'id_field': 'ID',
                'title_field': 'title',
                'year_field': 'year',
                'creator_field': 'publisher'
            }
        }

        self.fuzzy_threshold = 90

    def normalize_text(self, text):
        if isinstance(text, list):
            text = ' '.join(str(item) for item in text)
        if isinstance(text, str):
            return text.strip().lower()
        return ''

    def is_duplicate(self, new_rec, existing_recs, config):
        id_field = config['id_field']
        title_field = config['title_field']
        year_field = config['year_field']
        creator_field = config['creator_field']

        def normalize_id(val):
            return str(val).replace('-', '').strip().lower() if val else ''

        new_id = normalize_id(new_rec.get(id_field))
        new_title = self.normalize_text(new_rec.get(title_field))
        new_year = str(new_rec.get(year_field)).strip() if new_rec.get(year_field) else ''
        new_creator = self.normalize_text(new_rec.get(creator_field))

        for rec in existing_recs:
            existing_id = normalize_id(rec.get(id_field))
            if new_id and existing_id and new_id == existing_id:
                return True

            if not new_id or not existing_id or new_id != existing_id:
                existing_title = self.normalize_text(rec.get(title_field))
                existing_year = str(rec.get(year_field)).strip() if rec.get(year_field) else ''
                existing_creator = self.normalize_text(rec.get(creator_field))

                title_score = fuzz.token_set_ratio(new_title, existing_title)
                if (
                        title_score >= self.fuzzy_threshold
                        and new_creator == existing_creator
                        and new_year == existing_year
                ):
                    return True

        return False

    def extract_submission_recommendations(self, source_list):
        cleaned = []
        if source_list:
            for rec in source_list:
                if rec and isinstance(rec, dict):
                    cleaned.append(rec)
        return cleaned

    def extract_comment_recommendations(self, source_list, submission_recs, temp_seen, config, comment_id):
        recs = []
        duplicates = []
        for rec in source_list:
            if not rec or not isinstance(rec, dict):
                continue

            if self.is_duplicate(rec, submission_recs, config):
                duplicates.append(rec)  # Only duplicates of submission recs
            elif not self.is_duplicate(rec, temp_seen, config):
                cleaned = rec.copy()
                cleaned['id'] = comment_id
                recs.append(cleaned)
                temp_seen.append(cleaned)  # Avoid repeat across comments
            # else: duplicate of another comment – skip, but not stored
        return recs, duplicates

    def process_submission(self, submission):
        category = submission.get('link_flair_text')
        if category not in self.category_map:
            logging.warning(f"Unknown category: {category}, skipping...")
            return None

        config = self.category_map[category]
        submission_recs = self.extract_submission_recommendations(submission.get('submission_recommendations', []))

        unique_comment_recs = []
        duplicates = []
        temp_seen = submission_recs.copy()  # used for avoiding duplicate comment recs

        for comment in submission.get('comments', []):
            comment_recs = comment.get('recommendations', [])
            unique_recs, dups = self.extract_comment_recommendations(comment_recs, submission_recs, temp_seen, config, comment.get("id", None))
            unique_comment_recs.extend(unique_recs)
            duplicates.extend(dups)

        # Remove duplicates in duplicates_removed list
        unique_duplicates = []
        for dup in duplicates:
            if not any(self.is_duplicate(dup, [seen], config) for seen in unique_duplicates):
                unique_duplicates.append(dup)

        submission['final_recommendations'] = unique_comment_recs
        submission['duplicates'] = unique_duplicates
        return submission

    def run(self):
        count = 0
        for submission in self.input_collection.find():
            result = self.process_submission(submission)
            if result:
                self.output_collection.insert_one(result)
                count += 1
                if count % 100 == 0:
                    logging.info(f"Processed {count} submissions...")
        logging.info(f"Finished processing {count} submissions.")
