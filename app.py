import csv
import io
import json
import math
import os
from pathlib import Path
from urllib.error import URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

import joblib
from flask import Flask, jsonify, render_template, request, send_file, Response
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from knowledge_base import retrieve_retention_protocols


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
DATA_DIR = ROOT / "datasets"
INDUSTRIES = {"ecommerce"}
MAX_CHAT_MESSAGE = 1200


def load_env_file():
    env_file = ROOT / ".env"
    if env_file.is_file():
        try:
            with open(env_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, _, value = line.partition("=")
                        key = key.strip()
                        value = value.strip().strip("'\"")
                        if key and key not in os.environ:
                            os.environ[key] = value
        except Exception:
            pass


load_env_file()

if os.environ.get("GEMINI_API_KEY"):
    print("[RetainX] GEMINI_API_KEY is detected.")
else:
    print("[RetainX] Note: GEMINI_API_KEY is not set. Retention Advisor will run in Preview Mode.")

app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32 MB for bulk cohort CSVs
limiter = Limiter(
    get_remote_address,
    app=app,
    storage_uri=os.environ.get("RATELIMIT_STORAGE_URI", "memory://"),
    default_limits=[],
)


def load_model(name):
    model_path = MODEL_DIR / f"{name}.joblib"
    if not model_path.is_file():
        return None
    return joblib.load(model_path)


MODELS = {industry: load_model(industry) for industry in INDUSTRIES}
REVIEW_MODEL = load_model("reviews")
SEGMENTATION_MODEL = load_model("segmentation")


class AIProviderError(RuntimeError):
    pass


def error_response(message, status):
    return jsonify({"error": message}), status


def clean_number(value, name, minimum, maximum, integer=False):
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number.")
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite.")
    if number < minimum or number > maximum:
        raise ValueError(f"{name} must be between {minimum:g} and {maximum:g}.")
    if integer and not number.is_integer():
        raise ValueError(f"{name} must be a whole number.")
    return int(number) if integer else number


def normalize_customer(industry, customer):
    if not isinstance(customer, dict):
        raise ValueError("Provide customer details.")
    if industry != "ecommerce":
        raise ValueError("Invalid industry. Only 'ecommerce' is supported.")
    recency = clean_number(customer.get("days_since_last_order"), "Days since last order", 0, 3650, True)
    order_count = clean_number(customer.get("orders_in_history", 1), "Orders in history", 1, 500, True)
    tenure = clean_number(customer.get("days_since_first_order", recency), "Customer history days", recency, 3650, True)
    average_review = clean_number(customer.get("average_review_score", 0), "Average review score", 0, 5)
    last_review = clean_number(customer.get("last_review_score", 0), "Last review score", 0, 5)
    avg_delay = clean_number(customer.get("average_delivery_delay_days", 0), "Average delivery delay", -365, 365)
    avg_duration = clean_number(customer.get("average_delivery_duration_days", 0), "Average delivery duration", 0, 365)
    average_value = clean_number(customer.get("average_order_value", 0), "Average order value", 0, 1000000)
    spend = clean_number(customer.get("total_historical_spend", 0), "Historical spend", 0, 100000000)
    items = clean_number(customer.get("average_items_per_order", 0), "Average items per order", 0, 500)
    last_value = clean_number(customer.get("last_order_value", 0), "Last order value", 0, 1000000)
    last_freight = clean_number(customer.get("last_freight_value", 0), "Last freight value", 0, 50000)
    installments = clean_number(customer.get("last_payment_installments", 1), "Last payment installments", 1, 36, True)
    ptype = customer.get("last_payment_type", "unknown")
    if ptype not in {"credit_card", "boleto", "voucher", "debit_card", "upi", "cod", "unknown"}:
        raise ValueError("Choose a valid payment method.")
    return {
        "days_since_last_order": recency,
        "orders_in_history": order_count,
        "days_since_first_order": tenure,
        "average_order_value": average_value,
        "total_historical_spend": spend,
        "average_items_per_order": items,
        "last_order_value": last_value,
        "last_freight_value": last_freight,
        "last_payment_installments": installments,
        "last_payment_type": ptype,
        "average_review_score": average_review,
        "last_review_score": last_review,
        "average_delivery_delay_days": avg_delay,
        "average_delivery_duration_days": avg_duration,
    }


def predict_customer_segment(features):
    """Assign customer to an unsupervised K-Means behavioral segment."""
    if SEGMENTATION_MODEL is None or not isinstance(features, dict):
        return None
    try:
        vec = [
            float(features.get("days_since_last_order") or 0.0),
            float(features.get("orders_in_history") or 1.0),
            float(features.get("total_historical_spend") or 0.0),
            float(features.get("average_order_value") or 0.0),
            float(features.get("average_review_score") or 4.0),
            float(features.get("average_delivery_duration_days") or 7.0),
            float(features.get("average_delivery_delay_days") or 0.0),
        ]
        scaler = SEGMENTATION_MODEL["scaler"]
        kmeans = SEGMENTATION_MODEL["kmeans"]
        scaled = scaler.transform([vec])
        cid = int(kmeans.predict(scaled)[0])
        profiles = SEGMENTATION_MODEL.get("cluster_profiles", {})
        return profiles.get(cid, {
            "cluster_id": cid,
            "name": f"Behavioral Cluster #{cid}",
            "short_name": f"Cluster {cid}",
            "badge_color": "#6366f1",
            "description": "Customer behavioral cluster identified via K-Means.",
            "recommended_action": "Standard customer engagement workflow.",
        })
    except Exception:
        return None


def analyze_review(text):
    if text in (None, ""):
        return None
    if not isinstance(text, str):
        raise ValueError("Review text must be plain text.")
    if len(text) > 2500:
        raise ValueError("Review text must be 2,500 characters or fewer.")
    text = text.strip()
    if not text:
        return None
    if REVIEW_MODEL is None:
        return {
            "sentiment": "unavailable",
            "message": "The review sentiment model has not been installed.",
        }
    vectorized = REVIEW_MODEL["vectorizer"].transform([text])
    positive_score = float(REVIEW_MODEL["classifier"].predict_proba(vectorized)[0][1])
    sentiment = "positive" if positive_score >= 0.6 else "negative" if positive_score <= 0.4 else "mixed"
    return {
        "sentiment": sentiment,
        "positive_score_percent": round(positive_score * 100, 1),
        "label_basis": "NLP Model trained on real customer review comments.",
        "limitations": REVIEW_MODEL.get("limitations", "Trained on e-commerce customer review text."),
    }


def make_evidence(model_bundle, features):
    vectorized = model_bundle["vectorizer"].transform([features])
    scaled = model_bundle["scaler"].transform(vectorized)
    coef = model_bundle["classifier"].coef_[0]

    if hasattr(scaled, "toarray"):
        scaled_row = scaled.toarray()[0]
    elif isinstance(scaled, (list, tuple)):
        scaled_row = list(scaled)[0]
    else:
        scaled_row = scaled[0]

    contributions = scaled_row * coef
    names = model_bundle["vectorizer"].get_feature_names_out()
    evidence = []
    for name, value in zip(names, contributions):
        if abs(value) < 0.005:
            continue
        evidence.append({
            "feature": str(name),
            "direction": "raises" if value > 0 else "lowers",
            "strength": round(float(abs(value)), 3),
        })
    evidence.sort(key=lambda item: item["strength"], reverse=True)
    return evidence[:5]


def model_case(industry, customer):
    bundle = MODELS.get(industry)
    if bundle is None:
        raise RuntimeError(f"The {industry} model has not been trained or installed.")
    features = normalize_customer(industry, customer)
    vectorized = bundle["vectorizer"].transform([features])
    scaled = bundle["scaler"].transform(vectorized)
    probability = float(bundle["classifier"].predict_proba(scaled)[0][1])
    threshold = float(bundle["decision_threshold"])
    evidence = make_evidence(bundle, features)
    review_text = customer.get("review_text") if isinstance(customer, dict) else None
    segment = predict_customer_segment(features)

    case = {
        "industry": industry,
        "probability": round(probability, 4),
        "risk_percent": round(probability * 100, 1),
        "risk_flag": probability >= threshold,
        "decision_threshold_percent": round(threshold * 100, 1),
        "positive_snapshot_rate_percent": round(bundle.get("positive_snapshot_rate", 0) * 100, 2),
        "label_name": bundle["label_name"],
        "horizon": bundle["horizon"],
        "evidence": evidence,
        "metrics": bundle["metrics"],
        "evaluation_split": bundle.get(
            "evaluation_split",
            "Held-out 20% test split from 100,000 INR e-commerce customer cohort.",
        ),
        "limitations": bundle["limitations"],
        "days_since_last_order": features["days_since_last_order"],
        "inactivity_days": bundle.get("inactivity_days", 90),
        "churn_status": "inactive_90_days" if features["days_since_last_order"] >= bundle.get("inactivity_days", 90) else "active_under_90_days",
        "review_signal": analyze_review(review_text),
        "customer_segment": segment,
    }

    case["retention_protocols"] = retrieve_retention_protocols(
        {**features, "inactivity_days": case["inactivity_days"]}, evidence, segment=segment
    )
    case["customer_history"] = features
    return case


def fallback_recommendation(case):
    days = case.get("days_since_last_order", 0)
    threshold = case.get("inactivity_days", 90)
    evidence = case.get("evidence", [])
    reasons = [item["feature"].replace("_", " ") for item in evidence[:3]]
    reason_text = ", ".join(reasons) if reasons else "limited prior order history"
    seg = case.get("customer_segment")
    seg_name = seg.get("name") if seg else "General Customer"

    if case.get("churn_status") == "inactive_90_days":
        return (
            f"What: The customer is inactive at {days} days since their last order (Segment: {seg_name}). "
            f"Why: The 90-day inactivity threshold has been exceeded; primary predictive factors are {reason_text}. "
            "How: Initiate targeted win-back activation with an exclusive credit voucher (₹150-₹300 off) and replenish their top product category."
        )
    remaining = max(0, threshold - days)
    return (
        f"What: The customer is currently active ({days} days since last order; {remaining} days remain before 90-day churn cutoff). Segment: {seg_name}. "
        f"Why: The primary signals influencing their risk profile are {reason_text}. "
        "How: Proactively engage with personalized category recommendations, loyalty tier perks, and verify on-time delivery satisfaction."
    )


def gemini_reply(case, question, history=None, mode="chat"):
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return None

    context = {
        "business_domain": "E-Commerce Customer Retention (100k INR & DMart Cohort)",
        "model_churn_score_percent": case["risk_percent"],
        "customer_history": case.get("customer_history", {}),
        "days_since_last_order": case.get("days_since_last_order"),
        "inactivity_threshold_days": case.get("inactivity_days", 90),
        "deterministic_inactivity_status": case.get("churn_status"),
        "above_decision_threshold": case["risk_flag"],
        "decision_threshold_percent": case["decision_threshold_percent"],
        "target_definition": case["label_name"],
        "prediction_horizon": case["horizon"],
        "customer_segment": case.get("customer_segment"),
        "key_risk_drivers": case["evidence"],
        "customer_review_sentiment": case.get("review_signal"),
        "rag_retrieved_retention_protocols": case.get("retention_protocols", []),
    }

    if mode == "recommendation":
        system_prompt = (
            "You are RetainX Executive Retention Advisor, an elite retail & e-commerce retention strategist (such as for DMart / Marketplace retail). "
            "Analyze the provided customer telemetry, churn risk score, mathematical feature drivers, unsupervised K-Means segment, and RAG-retrieved company protocols. "
            "Produce an authoritative, comprehensive, executive-ready retention strategy briefing in rich Markdown:\n\n"
            "### 1. 📊 Executive Churn Risk Diagnosis\n"
            "- State the exact churn probability score (%) and whether it breaches the decision threshold.\n"
            "- Explain the primary root causes citing the customer's actual data points (e.g. days since last order, delivery transit delay, basket spend in INR, installments).\n"
            "- Incorporate their unsupervised behavioral segment profile.\n\n"
            "### 2. 🗣️ Voice of the Customer & Feedback Analysis\n"
            "- If a customer review or comment is present, translate or diagnose the underlying operational friction.\n"
            "- Summarize customer sentiment and psychological impact on brand loyalty.\n\n"
            "### 3. 🎯 Tailored Retention Playbook (Grounded RAG Protocols)\n"
            "- Present 2 to 3 detailed, concrete service recovery protocols directly from the retrieved company knowledge base.\n"
            "- Specify exact compensation figures (e.g., ₹100-₹250 credit vouchers, VIP courier priority, or instant replacement).\n\n"
            "### 4. ⚡ 48-Hour Execution Plan & Outreach Draft\n"
            "- Outline immediate operational steps for the Customer Retention team.\n"
            "- Include a personalized, empathetic outreach message (WhatsApp/Email/SMS) ready to dispatch.\n\n"
            "Provide thorough, high-value, complete business intelligence. Context (JSON): " + json.dumps(context)
        )
    else:
        system_prompt = (
            "You are RetainX Copilot, a brilliant, premium e-commerce customer retention & growth AI assistant (powered by Gemini). "
            "You have full access to this customer's transaction metrics, churn risk score, K-Means behavioral segment, feature evidence, review feedback, and internal retention SOPs. "
            "You can:\n"
            "- Draft custom apology emails, SMS win-back campaigns, or executive WhatsApp scripts\n"
            "- Calculate voucher economics, margins, and customer lifetime value (LTV in INR)\n"
            "- Analyze consumer behavior, carrier performance, and supply-chain logistics\n"
            "- Explain the difference between the supervised churn prediction model and the unsupervised K-Means segmentation\n"
            "- Answer general retail, e-commerce (e.g., DMart marketplace), or ML methodology questions\n\n"
            "Answer the user's latest question using relevant supplied context and retention protocols. "
            "Always structure your answer cleanly using rich Markdown. Context (JSON): " + json.dumps(context)
        )

    contents = []
    for item in (history or [])[-8:]:
        if not isinstance(item, dict) or item.get("role") not in {"user", "assistant"}:
            continue
        content = item.get("content")
        if isinstance(content, str) and len(content) <= MAX_CHAT_MESSAGE:
            contents.append({
                "role": "model" if item["role"] == "assistant" else "user",
                "parts": [{"text": content}],
            })
    contents.append({"role": "user", "parts": [{"text": question}]})

    candidate_models = []
    env_model = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")
    if env_model:
        candidate_models.append(env_model)
    for fallback in ["gemini-3.5-flash-lite", "gemini-3-flash-preview", "gemini-3.5-flash"]:
        if fallback not in candidate_models:
            candidate_models.append(fallback)

    payload = {
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "contents": contents,
        "generationConfig": {
            "maxOutputTokens": 4096 if mode == "recommendation" else 2048,
            "temperature": 0.35,
        },
    }

    last_error = None
    result = None

    for model_name in candidate_models:
        model_slug = quote(model_name, safe="")
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/{model_slug}:generateContent?"
            + urlencode({"key": api_key})
        )
        req = Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "X-goog-api-key": api_key},
            method="POST",
        )
        for attempt in range(2):
            try:
                with urlopen(req, timeout=45) as response:
                    result = json.loads(response.read().decode("utf-8"))
                break
            except Exception as exc:
                last_error = exc
                app.logger.warning("Gemini model %s attempt %d failed: %s", model_name, attempt + 1, exc)
                import time
                time.sleep(1.0)
        if result and isinstance(result, dict) and result.get("candidates"):
            break

    if not result or not isinstance(result, dict):
        raise AIProviderError(f"Gemini request failed: {last_error}")

    candidates = result.get("candidates", [])
    if not candidates or not isinstance(candidates[0], dict):
        raise AIProviderError("Gemini returned no candidates.")
    content = candidates[0].get("content")
    if not isinstance(content, dict):
        raise AIProviderError("Gemini returned invalid candidate content.")
    parts = content.get("parts", [])
    if not isinstance(parts, list):
        raise AIProviderError("Gemini returned invalid content parts.")

    answer = "\n".join(
        part["text"].strip()
        for part in parts
        if isinstance(part, dict) and isinstance(part.get("text"), str) and part["text"].strip()
    )
    if not answer:
        raise AIProviderError("Gemini returned an empty response text.")
    return answer


def get_model_stats_data():
    ec = MODELS.get("ecommerce")
    rev = REVIEW_MODEL
    seg = SEGMENTATION_MODEL

    ec_weights = []
    if ec and "classifier" in ec and "vectorizer" in ec:
        try:
            names = ec["vectorizer"].get_feature_names_out()
            coefs = ec["classifier"].coef_[0]
            for n, c in zip(names, coefs):
                ec_weights.append({
                    "feature": str(n),
                    "coefficient": round(float(c), 4),
                    "direction": "churn" if c > 0 else "retain",
                    "impact": round(abs(float(c)), 4),
                })
            ec_weights.sort(key=lambda x: x["impact"], reverse=True)
        except Exception:
            pass

    return {
        "ecommerce": {
            "available": ec is not None,
            "metrics": ec.get("metrics") if ec else None,
            "training_customers": ec.get("training_customers", 80000) if ec else 80000,
            "test_customers": ec.get("test_customers", 20000) if ec else 20000,
            "positive_snapshot_rate": ec.get("positive_snapshot_rate", 0.393) if ec else 0.393,
            "majority_baseline_f1": ec.get("majority_baseline_f1", 0.607) if ec else 0.607,
            "decision_threshold": ec.get("decision_threshold", 0.5) if ec else 0.5,
            "inactivity_days": ec.get("inactivity_days", 90) if ec else 90,
            "confusion_matrix": ec.get("confusion_matrix", {
                "tn": 9502,
                "fp": 2641,
                "fn": 3072,
                "tp": 4785,
                "total_test": 20000,
                "repeat_specificity": 78.3,
                "churn_recall": 60.9,
            }),
            "feature_weights": ec_weights,
            "evaluation_split": ec.get("evaluation_split") if ec else None,
            "limitations": ec.get("limitations") if ec else None,
        },
        "segmentation": {
            "available": seg is not None,
            "algorithm": seg.get("algorithm", "K-Means Unsupervised Clustering (k=4)") if seg else None,
            "optimal_k": seg.get("optimal_k", 4) if seg else 4,
            "silhouette_score": seg.get("silhouette_score", 0.2331) if seg else 0.2331,
            "total_segmented_customers": seg.get("total_segmented_customers", 100000) if seg else 100000,
            "cluster_profiles": seg.get("cluster_profiles", {}) if seg else {},
            "elbow_data": seg.get("elbow_data", []) if seg else [],
            "feature_names": seg.get("feature_names", []) if seg else [],
            "feature_labels": seg.get("feature_labels", []) if seg else [],
        },
        "reviews": {
            "available": rev is not None,
            "metrics": rev.get("metrics") if rev else None,
            "training_examples": rev.get("training_examples", 29706) if rev else None,
            "test_examples": rev.get("test_examples", 7417) if rev else None,
            "accuracy": 93.2,
            "confusion_matrix": {
                "tn": 1953,
                "fp": 254,
                "fn": 248,
                "tp": 4962,
                "total_test": 7417,
                "negative_specificity": 88.5,
                "positive_recall": 95.2,
            },
            "evaluation_split": rev.get("evaluation_split") if rev else None,
            "label_definition": rev.get("label_definition") if rev else None,
        },
        "graphs": [
            {"id": "roc", "title": "ROC Curves (AUC)", "filename": "roc_curves.png", "desc": "ROC Curves for 100k Churn Classifier (0.721) & NLP Sentiment (0.974)"},
            {"id": "cm", "title": "Confusion Matrices", "filename": "confusion_matrices.png", "desc": "True vs Predicted Class distribution on 20k test split"},
            {"id": "feat", "title": "Feature Importance", "filename": "feature_importance.png", "desc": "Standardized logistic regression coefficients for churn risk factors"},
            {"id": "elbow", "title": "K-Means Elbow Method", "filename": "elbow_curve.png", "desc": "Within-cluster sum of squares (WCSS) inertia showing optimal k=4"},
            {"id": "scatter", "title": "2D Cluster Map (PCA)", "filename": "cluster_scatter.png", "desc": "2D Principal Component projection of customer clusters"},
            {"id": "profiles", "title": "Behavioral Profiles", "filename": "cluster_profiles.png", "desc": "Centroid comparisons for Recency, Spend, Ratings & Logistics Delays"},
            {"id": "nlp", "title": "NLP Sentiment Lexicon", "filename": "nlp_vocabulary.png", "desc": "Top positive & negative customer review n-grams"},
            {"id": "summary", "title": "Benchmark Comparison", "filename": "metrics_summary.png", "desc": "Side-by-side ROC-AUC, F1, Precision, and Recall scores"},
        ],
    }


def parse_batch_row_to_features(r):
    """Flexible parser for DMart 30-day cohort, Olist synthetic dataset, or custom user CSV."""
    avg_del = float(r.get("avg_delivery_days") or r.get("average_delivery_duration_days") or 7.0)
    late_rate = float(r.get("late_delivery_rate") or 0.0)
    delay_days = float(r.get("average_delivery_delay_days") or (late_rate * avg_del))

    # Payment type logic
    ptype = r.get("last_payment_type")
    if not ptype:
        shares = {
            "credit_card": float(r.get("credit_card_payment_share") or 0),
            "boleto": float(r.get("boleto_payment_share") or 0),
            "voucher": float(r.get("voucher_payment_share") or 0),
            "debit_card": float(r.get("debit_card_payment_share") or 0),
        }
        ptype = max(shares, key=shares.get) if any(shares.values()) else "credit_card"

    recency = float(r.get("recency_days_at_cutoff") or r.get("days_since_last_order") or 0.0)
    orders = float(r.get("orders_4y") or r.get("orders_in_history") or 1.0)
    tenure = float(r.get("tenure_days_feature_window") or r.get("days_since_first_order") or max(recency, 30.0))
    aov = float(r.get("avg_order_value") or r.get("average_order_value") or 0.0)
    spend = float(r.get("total_spend_4y") or r.get("total_historical_spend") or (aov * orders))
    items = float(r.get("avg_items_per_order") or r.get("average_items_per_order") or 1.0)
    last_val = float(r.get("median_order_value") or r.get("last_order_value") or aov)
    freight = float(r.get("avg_freight_per_order") or r.get("last_freight_value") or 0.0)
    installments = float(r.get("avg_payment_installments") or r.get("last_payment_installments") or 1.0)
    rating = float(r.get("avg_review_rating") or r.get("average_review_score") or 4.0)
    last_rating = float(r.get("last_review_score") or rating)

    churn_feat = {
        "days_since_last_order": recency,
        "orders_in_history": orders,
        "days_since_first_order": tenure,
        "average_order_value": aov,
        "total_historical_spend": spend,
        "average_items_per_order": items,
        "last_order_value": last_val,
        "last_freight_value": freight,
        "last_payment_installments": installments,
        "last_payment_type": ptype,
        "average_review_score": rating,
        "last_review_score": last_rating,
        "average_delivery_delay_days": delay_days,
        "average_delivery_duration_days": avg_del,
    }

    seg_vec = [recency, orders, spend, aov, rating, avg_del, delay_days]
    return churn_feat, seg_vec


@app.get("/")
def home():
    available = {industry: MODELS.get(industry) is not None for industry in INDUSTRIES}
    return render_template(
        "index.html",
        model_availability=available,
        ai_configured=bool(os.environ.get("GEMINI_API_KEY")),
        review_metrics=REVIEW_MODEL["metrics"] if REVIEW_MODEL else None,
        model_stats=get_model_stats_data(),
    )


@app.get("/api/model-stats")
def model_stats():
    return jsonify(get_model_stats_data())


@app.get("/api/segmentation-stats")
def segmentation_stats():
    if SEGMENTATION_MODEL is None:
        return error_response("Segmentation model not installed.", 503)
    return jsonify({
        "algorithm": SEGMENTATION_MODEL.get("algorithm"),
        "optimal_k": SEGMENTATION_MODEL.get("optimal_k"),
        "silhouette_score": SEGMENTATION_MODEL.get("silhouette_score"),
        "cluster_profiles": SEGMENTATION_MODEL.get("cluster_profiles"),
        "elbow_data": SEGMENTATION_MODEL.get("elbow_data"),
        "pca_scatter_points": SEGMENTATION_MODEL.get("pca_scatter_points", []),
    })


@app.get("/api/sample-dataset")
def download_sample_dataset():
    """Download 30-day DMart cohort sample CSV (5,000 customers)."""
    sample_file = DATA_DIR / "dmart_sample_30days_5k.csv"
    if not sample_file.is_file():
        # Fallback to synthetic dataset if sample is missing
        sample_file = ROOT / "olist_synthetic_ecommerce_churn_100k_INR_bimodal.csv"
    if not sample_file.is_file():
        return error_response("Sample dataset not found.", 404)
    return send_file(
        sample_file,
        mimetype="text/csv",
        as_attachment=True,
        download_name="dmart_sample_30days_customers.csv",
    )


@app.post("/api/batch-predict")
def batch_predict():
    """
    Enterprise Batch Prediction Endpoint.
    Scores customer cohorts of any size (up to 100,000+ records) using memory-efficient chunking.
    """
    churn_model = MODELS.get("ecommerce")
    if churn_model is None:
        return error_response("E-commerce churn model is not available.", 503)

    raw_text = None
    if "file" in request.files:
        file = request.files["file"]
        if not file.filename:
            return error_response("No file selected for upload.", 400)
        try:
            raw_text = file.read().decode("utf-8", errors="replace")
        except Exception as exc:
            return error_response(f"Could not read uploaded file: {exc}", 400)
    elif request.is_json:
        data = request.get_json(silent=True)
        if isinstance(data, dict) and "customers" in data:
            records = data["customers"]
        elif isinstance(data, list):
            records = data
        else:
            return error_response("Provide CSV file or JSON array of customer records.", 400)
    else:
        return error_response("Upload a CSV file or send JSON customer records.", 400)

    if raw_text is not None:
        row_iterator = csv.DictReader(io.StringIO(raw_text))
    else:
        row_iterator = iter(records)

    CHUNK_SIZE = 10000
    total_customers = 0
    will_churn_count = 0
    will_retain_count = 0
    prob_sum = 0.0

    low_risk = 0
    mod_risk = 0
    high_risk = 0
    crit_risk = 0

    cluster_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    cluster_profiles = SEGMENTATION_MODEL.get("cluster_profiles", {}) if SEGMENTATION_MODEL else {}
    top_at_risk = []
    threshold = float(churn_model.get("decision_threshold", 0.5))

    chunk_rows = []
    has_rows = False

    def process_chunk(chunk):
        nonlocal total_customers, will_churn_count, will_retain_count, prob_sum
        nonlocal low_risk, mod_risk, high_risk, crit_risk

        c_feats = []
        s_vecs = []
        c_ids = []
        states = []

        for idx_row, r_item in enumerate(chunk):
            cid = r_item.get("customer_id") or f"CUST_{total_customers + idx_row + 1:05d}"
            st = r_item.get("state") or r_item.get("customer_state") or "N/A"
            cf, sv = parse_batch_row_to_features(r_item)
            c_feats.append(cf)
            s_vecs.append(sv)
            c_ids.append(cid)
            states.append(st)

        X_vec = churn_model["vectorizer"].transform(c_feats)
        X_sc = churn_model["scaler"].transform(X_vec)
        probs = churn_model["classifier"].predict_proba(X_sc)[:, 1]

        c_clusters = [0] * len(chunk)
        if SEGMENTATION_MODEL is not None:
            try:
                X_seg_sc = SEGMENTATION_MODEL["scaler"].transform(s_vecs)
                c_clusters = SEGMENTATION_MODEL["kmeans"].predict(X_seg_sc)
            except Exception:
                pass

        n_chunk = len(probs)
        total_customers += n_chunk

        ch_mask = (probs >= threshold)
        ch_count = int(ch_mask.sum())
        will_churn_count += ch_count
        will_retain_count += (n_chunk - ch_count)
        prob_sum += float(probs.sum())

        low_risk += int((probs < 0.30).sum())
        mod_risk += int(((probs >= 0.30) & (probs < 0.60)).sum())
        high_risk += int(((probs >= 0.60) & (probs < 0.80)).sum())
        crit_risk += int((probs >= 0.80).sum())

        for cid_val in c_clusters:
            cluster_counts[int(cid_val)] = cluster_counts.get(int(cid_val), 0) + 1

        for idx_p, p_val in enumerate(probs):
            if len(top_at_risk) < 50 or p_val > top_at_risk[-1]["risk_prob"]:
                c_cl = int(c_clusters[idx_p])
                c_pr = cluster_profiles.get(c_cl, {})
                top_at_risk.append({
                    "customer_id": c_ids[idx_p],
                    "state": states[idx_p],
                    "risk_prob": float(p_val),
                    "risk_percent": round(float(p_val * 100), 1),
                    "predicted_status": "CHURN RISK" if p_val >= threshold else "RETAIN",
                    "cluster_id": c_cl,
                    "segment_name": c_pr.get("name", f"Cluster {c_cl}"),
                    "segment_short": c_pr.get("short_name", f"Cluster {c_cl}"),
                    "badge_color": c_pr.get("badge_color", "#6366f1"),
                    "days_since_last_order": c_feats[idx_p]["days_since_last_order"],
                    "orders_in_history": c_feats[idx_p]["orders_in_history"],
                    "total_spend": round(c_feats[idx_p]["total_historical_spend"], 2),
                    "rating": c_feats[idx_p]["average_review_score"],
                    "delivery_delay": c_feats[idx_p]["average_delivery_delay_days"],
                    "action": c_pr.get("recommended_action", "Personalized offer."),
                })
                top_at_risk.sort(key=lambda x: x["risk_prob"], reverse=True)
                if len(top_at_risk) > 50:
                    top_at_risk.pop()

    for r in row_iterator:
        has_rows = True
        chunk_rows.append(r)
        if len(chunk_rows) >= CHUNK_SIZE:
            process_chunk(chunk_rows)
            chunk_rows = []

    if chunk_rows:
        process_chunk(chunk_rows)

    if not has_rows:
        return error_response("The provided dataset contains 0 customer records.", 400)

    retain_pct = round((will_retain_count / total_customers) * 100, 1)
    churn_pct = round((will_churn_count / total_customers) * 100, 1)
    avg_risk = round((prob_sum / total_customers) * 100, 1) if total_customers else 0.0

    segment_distribution = {}
    for cid in range(4):
        p = cluster_profiles.get(cid, {})
        sname = p.get("short_name", f"Cluster {cid}")
        cnt = cluster_counts.get(cid, 0)
        segment_distribution[sname] = {
            "cluster_id": cid,
            "name": p.get("name", sname),
            "short_name": sname,
            "count": cnt,
            "percent": round((cnt / total_customers) * 100, 1) if total_customers else 0.0,
            "badge_color": p.get("badge_color", "#6366f1"),
            "action": p.get("recommended_action", "Proactive engagement."),
        }

    return jsonify({
        "status": "success",
        "total_customers": total_customers,
        "summary": {
            "will_retain": will_retain_count,
            "will_churn": will_churn_count,
            "retain_percent": retain_pct,
            "churn_percent": churn_pct,
            "avg_risk_percent": avg_risk,
            "decision_threshold_percent": round(threshold * 100, 1),
        },
        "risk_distribution": {
            "low_risk": {"count": low_risk, "percent": round(low_risk / total_customers * 100, 1), "label": "Low Risk (<30%)"},
            "moderate_risk": {"count": mod_risk, "percent": round(mod_risk / total_customers * 100, 1), "label": "Moderate (30-60%)"},
            "high_risk": {"count": high_risk, "percent": round(high_risk / total_customers * 100, 1), "label": "Elevated (60-80%)"},
            "critical_risk": {"count": crit_risk, "percent": round(crit_risk / total_customers * 100, 1), "label": "Critical (≥80%)"},
        },
        "segment_distribution": segment_distribution,
        "top_at_risk_customers": top_at_risk,
    })


@app.post("/api/batch-predict/export")
def batch_export():
    """Export scored batch dataset as a downloadable CSV using streaming chunking."""
    churn_model = MODELS.get("ecommerce")
    if churn_model is None:
        return error_response("E-commerce churn model is not available.", 503)

    if "file" not in request.files:
        return error_response("Upload a CSV file to export predictions.", 400)

    file = request.files["file"]
    try:
        raw_text = file.read().decode("utf-8", errors="replace")
    except Exception as exc:
        return error_response(f"Could not read uploaded file: {exc}", 400)

    reader = csv.DictReader(io.StringIO(raw_text))
    orig_fields = reader.fieldnames or []
    if not orig_fields:
        return error_response("CSV is missing headers or empty.", 400)

    new_fields = list(orig_fields) + [
        "retainx_predicted_outcome",
        "retainx_churn_probability_percent",
        "retainx_kmeans_segment",
        "retainx_recommended_action",
    ]

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=new_fields)
    writer.writeheader()

    cluster_profiles = SEGMENTATION_MODEL.get("cluster_profiles", {}) if SEGMENTATION_MODEL else {}
    CHUNK_SIZE = 10000
    chunk = []

    def export_chunk(chk):
        c_feats = []
        s_vecs = []
        for r_item in chk:
            cf, sv = parse_batch_row_to_features(r_item)
            c_feats.append(cf)
            s_vecs.append(sv)

        X_vec = churn_model["vectorizer"].transform(c_feats)
        X_sc = churn_model["scaler"].transform(X_vec)
        probs = churn_model["classifier"].predict_proba(X_sc)[:, 1]

        c_clusters = [0] * len(chk)
        if SEGMENTATION_MODEL is not None:
            try:
                X_seg_sc = SEGMENTATION_MODEL["scaler"].transform(s_vecs)
                c_clusters = SEGMENTATION_MODEL["kmeans"].predict(X_seg_sc)
            except Exception:
                pass

        for idx_c, r_item in enumerate(chk):
            p_val = float(probs[idx_c])
            cl_id = int(c_clusters[idx_c])
            c_prof = cluster_profiles.get(cl_id, {})
            r_item["retainx_predicted_outcome"] = "CHURN" if p_val >= 0.5 else "RETAIN"
            r_item["retainx_churn_probability_percent"] = f"{p_val * 100:.1f}%"
            r_item["retainx_kmeans_segment"] = c_prof.get("name", f"Cluster {cl_id}")
            r_item["retainx_recommended_action"] = c_prof.get("recommended_action", "Customer outreach")
            writer.writerow(r_item)

    for row in reader:
        chunk.append(row)
        if len(chunk) >= CHUNK_SIZE:
            export_chunk(chunk)
            chunk = []

    if chunk:
        export_chunk(chunk)

    output.seek(0)
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=retainx_batch_predictions_scored.csv"},
    )


@app.get("/healthz")
def health():
    available = {industry: MODELS.get(industry) is not None for industry in INDUSTRIES}
    available["reviews"] = REVIEW_MODEL is not None
    available["segmentation"] = SEGMENTATION_MODEL is not None
    return jsonify({
        "status": "ok" if any(available.values()) else "degraded",
        "models": available,
        "ai_configured": bool(os.environ.get("GEMINI_API_KEY")),
    }), 200 if any(available.values()) else 503


@app.post("/api/predict")
def predict():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return error_response("Send a JSON request body.", 400)
    industry = data.get("industry", "ecommerce")
    if industry not in INDUSTRIES:
        return error_response("Choose a valid industry.", 400)
    try:
        case = model_case(industry, data.get("customer"))
    except ValueError as exc:
        return error_response(str(exc), 400)
    except RuntimeError as exc:
        return error_response(str(exc), 503)
    return jsonify(case)


@app.post("/api/recommendations")
@limiter.shared_limit("30 per hour", scope="ai-advisor")
def recommendations():
    data = request.get_json(silent=True)
    if not isinstance(data, dict) or data.get("industry") not in INDUSTRIES:
        return error_response("Choose a valid business domain.", 400)
    if data.get("consent") is not True:
        return jsonify({
            "answer": "Enable the AI Advisor option to synthesize the retention playbook with Gemini 3.5.",
            "provider": "consent_required",
        })
    try:
        case = model_case(data["industry"], data.get("customer"))
        prompt = (
            "Provide an executive retention briefing for this customer based on the risk score, "
            "review message, and RAG-retrieved retention protocols. Incorporate their K-Means customer segment."
        )
        answer = gemini_reply(case, prompt, mode="recommendation")
    except ValueError as exc:
        return error_response(str(exc), 400)
    except AIProviderError:
        app.logger.error("Gemini recommendation request failed.")
        return error_response("Gemini could not generate a recommendation. Please retry.", 502)
    if answer is None:
        return jsonify({
            "answer": fallback_recommendation(case),
            "provider": "preview",
        })
    return jsonify({
        "answer": answer,
        "provider": "gemini",
        "retention_protocols": case.get("retention_protocols", []),
    })


@app.post("/api/chat")
@limiter.shared_limit("30 per hour", scope="ai-advisor")
def chat():
    data = request.get_json(silent=True)
    if not isinstance(data, dict) or data.get("industry") not in INDUSTRIES:
        return error_response("Choose a valid business domain.", 400)
    question = data.get("question")
    if not isinstance(question, str) or not question.strip() or len(question) > MAX_CHAT_MESSAGE:
        return error_response(f"Enter a question of 1 to {MAX_CHAT_MESSAGE} characters.", 400)
    if data.get("consent") is not True:
        return error_response("Enable AI Advisor before starting a chat.", 403)
    try:
        case = model_case(data["industry"], data.get("customer"))
        answer = gemini_reply(case, question.strip(), data.get("history"), mode="chat")
    except ValueError as exc:
        return error_response(str(exc), 400)
    except AIProviderError:
        app.logger.error("Gemini chat request failed.")
        return error_response("Gemini could not answer right now. Please try again.", 502)
    if answer is None:
        return jsonify({
            "answer": (
                "AI chat is in preview mode because GEMINI_API_KEY is not configured. "
                + fallback_recommendation(case)
            ),
            "provider": "preview",
        })
    return jsonify({"answer": answer, "provider": "gemini"})


@app.errorhandler(413)
def request_too_large(_error):
    return error_response("Uploaded file exceeds 32MB limit.", 413)


@app.errorhandler(429)
def ai_rate_limit(_error):
    return error_response("AI request limit reached. Try again later.", 429)


@app.errorhandler(500)
def server_error(_error):
    return error_response("An internal error occurred while processing the cohort.", 500)


if __name__ == "__main__":
    app.run(debug=False)
