import os
import time
import json
import re
from datetime import datetime, timedelta

from flask import Flask, request, jsonify, render_template, abort
from pymongo import MongoClient
from dotenv import load_dotenv

from hf_helper import call_qwen_inference, build_prompt

load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI")
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "adminpass")

if not MONGODB_URI:
    raise RuntimeError("MONGODB_URI not set in environment")

if not HF_API_TOKEN:
    print("Warning: HF_API_TOKEN not set. /review-code will fail until set.")

app = Flask(__name__, template_folder="templates")

client = MongoClient(MONGODB_URI)
db = client.codereview_ai
reviews_col = db.code_reviews


@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({"status": "ok", "time": datetime.utcnow().isoformat()})


@app.route("/review-code", methods=["POST"])
def review_code():
    payload = request.get_json(force=True)
    code = payload.get("code")
    language = payload.get("language", "auto")
    session_id = payload.get("session_id")
    email = payload.get("email")

    if not code:
        return jsonify({"success": False, "error": "code is required"}), 400

    start_time = time.time()
    prompt = build_prompt(code=code, language=language)

    api_start = time.time()
    model_text = call_qwen_inference(prompt)
    api_latency = time.time() - api_start

    parsed = None
    try:
        parsed = json.loads(model_text)
    except Exception:
        m = re.search(r"(\{[\s\S]*\})", model_text)
        if m:
            try:
                parsed = json.loads(m.group(1))
            except Exception:
                parsed = {"raw": model_text}
        else:
            parsed = {"raw": model_text}

    review_doc = {
        "language": language,
        "code": code,
        "code_length": len(code),
        "session_id": session_id,
        "email": email,
        "model_raw_output": model_text,
        "parsed_output": parsed,
        "api_latency": api_latency,
        "response_time": time.time() - start_time,
        "created_at": datetime.utcnow(),
        "satisfaction": None,
        "feedback": None,
    }

    result = reviews_col.insert_one(review_doc)
    review_id = str(result.inserted_id)

    resp = {
        "success": True,
        "review_id": review_id,
        "report": parsed,
        "api_latency": api_latency,
        "response_time": review_doc["response_time"],
    }

    return jsonify(resp), 200


@app.route("/feedback", methods=["POST"])
def feedback():
    payload = request.get_json(force=True)
    review_id = payload.get("review_id")
    satisfaction = payload.get("user_satisfaction")
    feedback_text = payload.get("feedback")

    if not review_id:
        return jsonify({"success": False, "error": "review_id required"}), 400

    from bson import ObjectId

    try:
        oid = ObjectId(review_id)
    except Exception:
        return jsonify({"success": False, "error": "invalid review_id"}), 400

    update = {
        "$set": {
            "satisfaction": satisfaction,
            "feedback": feedback_text,
            "feedback_at": datetime.utcnow(),
        }
    }

    res = reviews_col.update_one({"_id": oid}, update)
    if res.matched_count == 0:
        return jsonify({"success": False, "error": "review not found"}), 404

    return jsonify({"success": True})


@app.route("/admin", methods=["GET"])
def admin_dashboard():
    pw = request.args.get("pw")
    if pw != ADMIN_PASSWORD:
        return abort(401)

    since = datetime.utcnow() - timedelta(days=1)
    total_reviews = reviews_col.count_documents({"created_at": {"$gte": since}})

    pipeline = [
        {"$match": {"created_at": {"$gte": since}}},
        {
            "$group": {
                "_id": None,
                "avg_response": {"$avg": "$response_time"},
                "avg_api_latency": {"$avg": "$api_latency"},
                "count": {"$sum": 1},
            }
        },
    ]
    agg = list(reviews_col.aggregate(pipeline))
    avg_response = agg[0]["avg_response"] if agg else 0
    avg_api_latency = agg[0]["avg_api_latency"] if agg else 0

    lang_pipeline = [
        {"$match": {"created_at": {"$gte": since}}},
        {"$group": {"_id": "$language", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
        {"$limit": 10},
    ]
    lang_groups = list(reviews_col.aggregate(lang_pipeline))

    issues_pipeline = [
        {
            "$match": {
                "created_at": {"$gte": since},
                "parsed_output.issues": {"$exists": True},
            }
        },
        {"$unwind": "$parsed_output.issues"},
        {"$group": {"_id": "$parsed_output.issues.issue", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
        {"$limit": 10},
    ]
    top_issues = list(reviews_col.aggregate(issues_pipeline))

    return render_template(
        "admin.html",
        total_reviews=total_reviews,
        avg_response=round(avg_response, 3),
        avg_api_latency=round(avg_api_latency, 3),
        lang_groups=lang_groups,
        top_issues=top_issues,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 3000)))
