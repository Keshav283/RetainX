import json
from urllib.request import Request, urlopen

payload = {
    "industry": "ecommerce",
    "consent": True,
    "customer": {
        "days_since_last_order": 120,
        "orders_in_history": 1,
        "days_since_first_order": 120,
        "average_order_value": 145.0,
        "total_historical_spend": 145.0,
        "average_items_per_order": 1,
        "last_order_value": 145.0,
        "last_freight_value": 38.5,
        "last_payment_installments": 1,
        "last_payment_type": "credit_card",
        "average_review_score": 1,
        "last_review_score": 1,
        "average_delivery_delay_days": 14.0,
        "average_delivery_duration_days": 28.0,
        "review_text": "O produto atrasou mais de 14 dias além do prazo prometido e a caixa chegou amassada e rasgada. Experiência péssima com o vendedor.",
    },
}

req = Request(
    "http://127.0.0.1:5000/api/recommendations",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)

try:
    with urlopen(req, timeout=60) as res:
        data = json.loads(res.read().decode("utf-8"))
        print("Provider:", data.get("provider"))
        print("Answer:\n", data.get("answer", ""))
except Exception as e:
    body = ""
    if hasattr(e, "read"):
        body = e.read().decode("utf-8", errors="replace")[:500]
    print("Error:", e, body)
