import csv
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "datasets"
DATA_DIR.mkdir(parents=True, exist_ok=True)
SAMPLE_CSV = DATA_DIR / "dmart_sample_30days_5k.csv"

random.seed(42)

REVIEW_TEMPLATES_POSITIVE = [
    "Groceries arrived on time and completely fresh. Excellent packaging!",
    "Great prices on household staples, delivery boy was polite and quick.",
    "Very satisfied with my DMart order. Everything was well-organized.",
    "Best online grocery experience. Arrived earlier than expected.",
    "Smooth checkout with UPI and instant doorstep delivery.",
]

REVIEW_TEMPLATES_NEUTRAL = [
    "Order delivered safely, but one item was replaced with an alternative brand.",
    "Decent service, took a bit longer than promised slot but acceptable.",
    "Good quality pulses and grains, wish delivery charge was free.",
]

REVIEW_TEMPLATES_NEGATIVE = [
    "Delivery was delayed by 3 days and customer care did not respond.",
    "Vegetables were bruised and detergent packet had leaked inside carton.",
    "Terrible logistics. Delivery person refused to climb stairs to 3rd floor.",
    "Items missing from my bag and still waiting for refund to credit.",
]

STATES = ["MH", "GJ", "KA", "DL", "TS", "TN", "UP", "RJ", "MP", "WB"]
PAYMENT_TYPES = ["upi", "credit_card", "debit_card", "cod", "voucher"]


def generate_dmart_sample(num_customers=5000):
    rows = []
    for i in range(1, num_customers + 1):
        cid = f"DMART_{i:04d}"
        state = random.choice(STATES)

        # Distribute into behavioral cohorts:
        # 60% active regular shoppers, 25% at-risk/slipping, 10% delivery delayed, 5% VIP bulk shoppers
        cohort = random.choices(["regular", "at_risk", "delayed", "vip"], weights=[0.60, 0.25, 0.10, 0.05])[0]

        if cohort == "vip":
            orders = random.randint(25, 60)
            tenure = random.randint(300, 730)
            recency = random.randint(1, 14)
            aov = round(random.uniform(4500, 15000), 2)
            spend = round(aov * orders, 2)
            items = random.randint(6, 16)
            last_order = round(aov * random.uniform(0.85, 1.25), 2)
            freight = 0.0  # VIP free delivery
            installments = random.choice([1, 1, 3, 6])
            ptype = random.choice(["credit_card", "upi"])
            rating = round(random.uniform(4.2, 5.0), 1)
            last_rating = random.choice([4, 5])
            delay = round(random.uniform(-2, 0.5), 1)
            duration = random.randint(1, 4)
            review = random.choice(REVIEW_TEMPLATES_POSITIVE)

        elif cohort == "delayed":
            orders = random.randint(2, 15)
            tenure = random.randint(60, 360)
            recency = random.randint(5, 28)
            aov = round(random.uniform(800, 3500), 2)
            spend = round(aov * orders, 2)
            items = random.randint(3, 8)
            last_order = round(aov * random.uniform(0.8, 1.2), 2)
            freight = round(random.uniform(49, 120), 2)
            installments = 1
            ptype = random.choice(["upi", "cod", "debit_card"])
            rating = round(random.uniform(1.0, 2.5), 1)
            last_rating = random.choice([1, 2])
            delay = round(random.uniform(3.5, 12.0), 1)
            duration = random.randint(8, 22)
            review = random.choice(REVIEW_TEMPLATES_NEGATIVE)

        elif cohort == "at_risk":
            orders = random.randint(1, 8)
            tenure = random.randint(90, 450)
            recency = random.randint(65, 130)  # High recency
            aov = round(random.uniform(500, 2200), 2)
            spend = round(aov * orders, 2)
            items = random.randint(2, 6)
            last_order = round(aov * random.uniform(0.7, 1.1), 2)
            freight = round(random.uniform(39, 79), 2)
            installments = 1
            ptype = random.choice(["cod", "upi", "debit_card"])
            rating = round(random.uniform(2.5, 3.8), 1)
            last_rating = random.choice([2, 3, 4])
            delay = round(random.uniform(0.5, 3.0), 1)
            duration = random.randint(3, 7)
            review = random.choice(REVIEW_TEMPLATES_NEUTRAL)

        else:  # regular active
            orders = random.randint(6, 30)
            tenure = random.randint(60, 500)
            recency = random.randint(2, 25)
            aov = round(random.uniform(1200, 4200), 2)
            spend = round(aov * orders, 2)
            items = random.randint(4, 10)
            last_order = round(aov * random.uniform(0.9, 1.15), 2)
            freight = round(random.uniform(0, 49), 2)
            installments = 1
            ptype = random.choice(["upi", "upi", "credit_card", "debit_card"])
            rating = round(random.uniform(4.0, 5.0), 1)
            last_rating = random.choice([4, 5])
            delay = round(random.uniform(-1.5, 1.0), 1)
            duration = random.randint(1, 5)
            review = random.choice(REVIEW_TEMPLATES_POSITIVE)

        rows.append({
            "customer_id": cid,
            "state": state,
            "days_since_last_order": recency,
            "orders_in_history": orders,
            "days_since_first_order": max(recency, tenure),
            "average_order_value": aov,
            "total_historical_spend": spend,
            "average_items_per_order": items,
            "last_order_value": last_order,
            "last_freight_value": freight,
            "last_payment_installments": installments,
            "last_payment_type": ptype,
            "average_review_score": rating,
            "last_review_score": last_rating,
            "average_delivery_delay_days": delay,
            "average_delivery_duration_days": duration,
            "review_text": review,
        })

    fieldnames = list(rows[0].keys())
    with open(SAMPLE_CSV, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Generated {len(rows):,} DMart 30-day sample customer records at {SAMPLE_CSV}")


if __name__ == "__main__":
    generate_dmart_sample()
