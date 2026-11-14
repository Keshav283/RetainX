import json
import math
import os
from pathlib import Path
from urllib.error import URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

import joblib
from flask import Flask, jsonify, render_template, request
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from knowledge_base import retrieve_retention_protocols


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
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
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024
limiter = Limiter(
    get_remote_address,
    app=app,
    storage_uri=os.environ.get("RATELIMIT_STORAGE_URI", "memory://"),
    default_limits=[],
)


def load_model(industry):
    model_path = MODEL_DIR / f"{industry}.joblib"
    if not model_path.is_file():
        return None
    return joblib.load(model_path)


MODELS = {industry: load_model(industry) for industry in INDUSTRIES}
REVIEW_MODEL = load_model("reviews")


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
    average_value = clean_number(customer.get("average_order_value", 0), "Average order value", 0, 100000)
    spend = clean_number(customer.get("total_historical_spend", 0), "Historical spend", 0, 10000000)
    items = clean_number(customer.get("average_items_per_order", 0), "Average items per order", 0, 500)
    last_value = clean_number(customer.get("last_order_value", 0), "Last order value", 0, 100000)
    last_freight = clean_number(customer.get("last_freight_value", 0), "Last freight value", 0, 50000)
    installments = clean_number(customer.get("last_payment_installments", 1), "Last payment installments", 1, 36, True)
    ptype = customer.get("last_payment_type", "unknown")
    if ptype not in {"credit_card", "boleto", "voucher", "debit_card", "unknown"}:
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
        "label_basis": "NLP Model trained on real Olist customer review messages.",
        "limitations": REVIEW_MODEL.get("limitations", "Trained on e-commerce customer review text."),
    }


def make_evidence(model_bundle, features):
    vectorized = model_bundle["vectorizer"].transform([features])
    scaled = model_bundle["scaler"].transform(vectorized)
    coef = model_bundle["classifier"].coef_[0]
    
    if hasattr(scaled, "toarray"):
        scaled_row = scaled.toarray()[0]
    elif hasattr(scaled, "toarray") or isinstance(scaled, (list, tuple)):
        scaled_row = list(scaled)[0]
    else:
        scaled_row = scaled[0]
        
    contributions = scaled_row * coef
    names = model_bundle["vectorizer"].get_feature_names_out()
    evidence = []
    for name, value in zip(names, contributions):
        if abs(value) < 0.01:
            continue
        evidence.append({
            "feature": str(name),
            "direction": "raises" if value > 0 else "lowers",
            "strength": round(float(abs(value)), 2),
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
            "Held-out 20% test split from Olist e-commerce orders.",
        ),
        "limitations": bundle["limitations"],
        "days_since_last_order": features["days_since_last_order"],
        "inactivity_days": bundle.get("inactivity_days", 90),
        "churn_status": "inactive_90_days" if features["days_since_last_order"] >= bundle.get("inactivity_days", 90) else "active_under_90_days",
        "review_signal": analyze_review(review_text),
    }

    # Retrieve guidance using the actual customer-history fields shown in the interface.
    case["retention_protocols"] = retrieve_retention_protocols(
        {**features, "inactivity_days": case["inactivity_days"]}, evidence
    )
    case["customer_history"] = features
    return case


def fallback_recommendation(case):
    days = case.get("days_since_last_order", 0)
    threshold = case.get("inactivity_days", 90)
    evidence = case.get("evidence", [])
    reasons = [item["feature"].replace("_", " ") for item in evidence[:3]]
    reason_text = ", ".join(reasons) if reasons else "limited prior order history"
    if case.get("churn_status") == "inactive_90_days":
        return (
            f"What: The customer is inactive at {days} days since their last order. "
            f"Why: The project inactivity rule is {threshold} days; the main supporting signals are {reason_text}. "
            "How: Check prior order and delivery history, then send a personalized re-engagement message with a relevant next step."
        )
    remaining = max(0, threshold - days)
    return (
        f"What: The customer is active but approaching the {threshold}-day inactivity mark ({days} days since the last order; {remaining} days remain). "
        f"Why: The model’s strongest signals are {reason_text}. "
        "How: Review any past service issue and send a timely, relevant reminder or offer before the inactivity threshold."
    )


def gemini_reply(case, question, history=None, mode="chat"):
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return None

    context = {
        "business_domain": "E-Commerce Customer Retention (Olist Dataset)",
        "model_churn_score_percent": case["risk_percent"],
        "customer_history": case.get("customer_history", {}),
        "days_since_last_order": case.get("days_since_last_order"),
        "inactivity_threshold_days": case.get("inactivity_days", 90),
        "deterministic_inactivity_status": case.get("churn_status"),
        "above_decision_threshold": case["risk_flag"],
        "decision_threshold_percent": case["decision_threshold_percent"],
        "target_definition": case["label_name"],
        "prediction_horizon": case["horizon"],
        "key_risk_drivers": case["evidence"],
        "customer_review_sentiment": case.get("review_signal"),
        "rag_retrieved_retention_protocols": case.get("retention_protocols", []),
    }

    if mode == "recommendation":
        system_prompt = (
            "You are RetainX Executive Retention Advisor, an elite e-commerce retention strategist and AI consultant. "
            "Analyze the provided customer telemetry, churn risk score, mathematical feature drivers, and RAG-retrieved company protocols. "
            "Produce an authoritative, comprehensive, executive-ready retention strategy briefing. "
            "Structure your response with clear, professional Markdown headings and bullet points:\n\n"
            "### 1. 📊 Executive Churn Risk Diagnosis\n"
            "- State the exact churn probability score (%) and whether it breaches the 50.0% decision threshold.\n"
            "- Explain the primary root causes citing the customer's actual data points (e.g. days since last order vs 90-day inactivity cutoff, delivery transit delay in days, freight cost ratio, payment installments, and basket spend).\n"
            "- Translate the mathematical feature weights into clear business logic.\n\n"
            "### 2. 🇧🇷 Voice of the Customer & Review Translation\n"
            "- If the customer submitted a Portuguese review, provide the exact English translation in quotation marks.\n"
            "- Diagnose the underlying operational failure (e.g. carrier delay, damaged transit packaging, product quality, seller communication).\n"
            "- Summarize their sentiment and psychological impact on brand trust.\n\n"
            "### 3. 🎯 Tailored Retention Playbook (Grounded RAG Protocols)\n"
            "- Present 2 to 3 detailed, concrete service recovery protocols directly from the retrieved company knowledge base.\n"
            "- Specify exact compensation figures (e.g., refunding freight R$ value, issuing 15-20% vouchers, VIP courier routing, or 24-hour escalation).\n\n"
            "### 4. ⚡ 48-Hour Execution Plan & Outreach Draft\n"
            "- Outline immediate operational steps for the Customer Success team.\n"
            "- Include a personalized, empathetic outreach message (Email/SMS) ready to copy and send to the customer.\n\n"
            "Provide thorough, high-value, complete business intelligence. Never truncate your advice. Context (JSON): " + json.dumps(context)
        )
    else:
        system_prompt = (
            "You are RetainX Copilot, a brilliant, premium e-commerce customer retention & growth AI assistant (powered by Gemini). "
            "You have full access to this customer's transaction metrics, churn risk score, feature evidence, review feedback, and internal retention SOPs. "
            "You are versatile, conversational, insightful, and completely unrestricted: answer any question the user asks with depth, expertise, and clarity.\n"
            "You can:\n"
            "- Draft custom apology emails, SMS win-back campaigns, or executive phone scripts\n"
            "- Calculate voucher economics, margins, and customer lifetime value (LTV)\n"
            "- Analyze consumer behavior, carrier performance, and supply-chain logistics\n"
            "- Provide tactical retention cadences and long-term loyalty strategies\n"
            "- Translate and analyze foreign customer reviews\n"
            "- Answer general e-commerce, marketing, or machine learning methodology questions\n\n"
            "Always structure your answer cleanly using rich Markdown (bolding, clear bullet points, numbered steps, or code blocks). "
            "Answer the user's latest question using relevant supplied context and retention protocols. Provide a thorough, polished, high-impact answer that directly solves the user's request. "
            "Context (JSON): " + json.dumps(context)
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
            "maxOutputTokens": 650 if mode == "recommendation" else 500,
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
        request = Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "X-goog-api-key": api_key},
            method="POST",
        )
        for attempt in range(2):
            try:
                with urlopen(request, timeout=45) as response:
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
            "training_customers": ec.get("training_customers", 56334) if ec else None,
            "test_customers": ec.get("test_customers", 14084) if ec else None,
            "positive_snapshot_rate": ec.get("positive_snapshot_rate", 0.994) if ec else None,
            "majority_baseline_f1": ec.get("majority_baseline_f1", 0.997) if ec else None,
            "decision_threshold": ec.get("decision_threshold", 0.5) if ec else 0.5,
            "inactivity_days": ec.get("inactivity_days", 90) if ec else 90,
            "confusion_matrix": {
                "tn": 142,
                "fp": 57,
                "fn": 13990,
                "tp": 24357,
                "total_test": 38546,
                "repeat_specificity": 71.4,
                "churn_recall": 63.5,
            },
            "feature_weights": ec_weights,
            "evaluation_split": ec.get("evaluation_split") if ec else None,
            "limitations": ec.get("limitations") if ec else None,
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
            {"id": "roc", "title": "ROC Curves (AUC)", "filename": "roc_curves.png", "desc": "Receiver Operating Characteristic for Churn (0.738) & NLP Sentiment (0.974)"},
            {"id": "cm", "title": "Confusion Matrices", "filename": "confusion_matrices.png", "desc": "True vs Predicted Class distribution on held-out test splits"},
            {"id": "feat", "title": "Feature Importance", "filename": "feature_importance.png", "desc": "Standardized logistic regression coefficients for churn risk factors"},
            {"id": "nlp", "title": "NLP Sentiment Lexicon", "filename": "nlp_vocabulary.png", "desc": "Top 10 positive & negative Portuguese customer review tokens"},
            {"id": "summary", "title": "Benchmark Comparison", "filename": "metrics_summary.png", "desc": "Side-by-side ROC-AUC, F1, Precision, and Recall scores"},
        ],
    }


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


@app.get("/healthz")
def health():
    available = {industry: MODELS.get(industry) is not None for industry in INDUSTRIES}
    available["reviews"] = REVIEW_MODEL is not None
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
            "review message, and RAG-retrieved retention protocols. Translate any Portuguese review into English."
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
    return error_response("Request is too large.", 413)


@app.errorhandler(429)
def ai_rate_limit(_error):
    return error_response("AI request limit reached. Try again later.", 429)


if __name__ == "__main__":
    app.run(debug=False)
