import json
import json_repair
from pymongo import MongoClient

BATCH_RESULTS_PATH = "result/result_2.jsonl"
client     = MongoClient("localhost", 27017)
db         = client["reddit_data"]
collection = db["[13]all_fields"]

with open(BATCH_RESULTS_PATH, "r", encoding="utf-8") as f:
    for line in f:
        result_obj = json.loads(line)
        custom_id  = result_obj.get("custom_id")
        if not custom_id:
            print("Skipping line with no custom_id.")
            continue

        doc = collection.find_one({"id": custom_id})
        if not doc:
            print(f"No MongoDB document found for id={custom_id}")
            continue

        raw  = result_obj["response"]["body"]["choices"][0]["message"]["content"]

        try:
            reply_json  = json_repair.loads(raw)
        except Exception as e:
            print(f"Parsing json_repair failed for {custom_id}: {e}")
            continue

        # Get all submission recommendations
        submission_recs = []

        # Make sure we have a sequence of dicts to iterate
        entries = reply_json if isinstance(reply_json, list) else [reply_json]

        for entry in entries:
            if not isinstance(entry, dict):
                continue
            for key, items in entry.items():
                if key.startswith("s_") and isinstance(items, list):
                    submission_recs.extend(items)

        # Get all comment recommendations
        comment_recs = {}

        # Ensure we have a list of dicts
        entries = reply_json if isinstance(reply_json, list) else [reply_json]

        for entry in entries:
            if not isinstance(entry, dict):
                continue
            for key, items in entry.items():
                if key.startswith("c_") and isinstance(items, list):
                    cm_id = key[2:]
                    comment_recs[cm_id] = items

        if submission_recs:
            collection.update_one(
                {"id": custom_id},
                {"$set": {"submission_recommendations": submission_recs}}
            )

        for cm_id, recs in comment_recs.items():
            res = collection.update_one(
                {"id": custom_id, "comments.id": cm_id},
                {"$set": {"comments.$.recommendations": recs}}
            )
            if not res.matched_count:
                print(f"Warning: no comment with id={cm_id} found in submission {custom_id}")
