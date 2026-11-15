import csv
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import joblib
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "datasets" / "olist"
SYNTHETIC_CSV = ROOT / "olist_synthetic_ecommerce_churn_100k_INR_bimodal.csv"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
RANDOM_SEED = 42
INACTIVITY_DAYS = 90
PREDICTION_WINDOW_DAYS = 90


def parse_date(value):
    value = (value or "").strip()
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None


def load_customer_orders():
    """Join Olist orders to stable customer IDs and historical order signals (kept for reference/tests)."""
    customer_ids = {}
    cust_file = DATA_DIR / "olist_customers_dataset.csv"
    if not cust_file.is_file():
        return {}

    with open(cust_file, encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            customer_ids[row["customer_id"]] = row["customer_unique_id"]

    item_summary = defaultdict(lambda: {"price": 0.0, "freight": 0.0, "items": 0})
    with open(DATA_DIR / "olist_order_items_dataset.csv", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            summary = item_summary[row["order_id"]]
            summary["price"] += float(row.get("price") or 0)
            summary["freight"] += float(row.get("freight_value") or 0)
            summary["items"] += 1

    payment_summary = defaultdict(lambda: {"value": 0.0, "installments": 1, "payment_type": "unknown"})
    with open(DATA_DIR / "olist_order_payments_dataset.csv", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            summary = payment_summary[row["order_id"]]
            summary["value"] += float(row.get("payment_value") or 0)
            summary["installments"] = max(summary["installments"], int(row.get("payment_installments") or 1))
            summary["payment_type"] = row.get("payment_type") or summary["payment_type"]

    review_summary = {}
    with open(DATA_DIR / "olist_order_reviews_dataset.csv", encoding="utf-8", errors="replace") as stream:
        for row in csv.DictReader(stream):
            try:
                review_summary[row["order_id"]] = int(row.get("review_score") or 0)
            except ValueError:
                continue

    orders = defaultdict(list)
    with open(DATA_DIR / "olist_orders_dataset.csv", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row.get("order_status") != "delivered":
                continue
            customer = customer_ids.get(row.get("customer_id"))
            purchased = parse_date(row.get("order_purchase_timestamp"))
            if not customer or not purchased or row.get("order_id") not in item_summary:
                continue
            order_id = row["order_id"]
            items = item_summary[order_id]
            payments = payment_summary[order_id]
            review_score = review_summary.get(order_id)
            delivered = parse_date(row.get("order_delivered_customer_date"))
            estimate = parse_date(row.get("order_estimated_delivery_date"))
            order_value = payments["value"] or items["price"] + items["freight"]
            orders[customer].append({
                "date": purchased,
                "order_value": order_value,
                "freight_value": items["freight"],
                "total_items": items["items"],
                "payment_installments": payments["installments"],
                "payment_type": payments["payment_type"],
                "review_score": review_score,
                "delivery_duration_days": max(0, (delivered - purchased).days) if delivered else None,
                "delivery_delay_days": (delivered - estimate).days if delivered and estimate else None,
            })

    for customer_orders in orders.values():
        customer_orders.sort(key=lambda order: order["date"])
    return orders


def make_snapshots(customer_orders, dataset_end, step_days=30):
    """Create customer snapshots (kept for reference and test suite compatibility)."""
    snapshots, labels, groups = [], [], []
    if not customer_orders:
        return snapshots, labels, groups
    start = min(order["date"] for rows in customer_orders.values() for order in rows)
    cutoff = dataset_end - timedelta(days=INACTIVITY_DAYS)
    snapshot_date = start + timedelta(days=INACTIVITY_DAYS - 1)

    while snapshot_date <= cutoff:
        for customer_id, orders in customer_orders.items():
            prior = [order for order in orders if order["date"] <= snapshot_date]
            if not prior:
                continue
            last_order = prior[-1]
            recency = (snapshot_date - last_order["date"]).days
            if recency >= INACTIVITY_DAYS:
                continue
            future_orders = [
                order for order in orders
                if snapshot_date < order["date"] <= snapshot_date + timedelta(days=PREDICTION_WINDOW_DAYS)
            ]
            latest = prior[-1]
            valid_reviews = [order["review_score"] for order in prior if order["review_score"]]
            delays = [order["delivery_delay_days"] for order in prior if order["delivery_delay_days"] is not None]
            durations = [order["delivery_duration_days"] for order in prior if order["delivery_duration_days"] is not None]
            total_spend = sum(order["order_value"] for order in prior)
            features = {
                "days_since_last_order": recency,
                "orders_in_history": len(prior),
                "days_since_first_order": (snapshot_date - prior[0]["date"]).days,
                "average_order_value": total_spend / len(prior),
                "total_historical_spend": total_spend,
                "average_items_per_order": sum(order["total_items"] for order in prior) / len(prior),
                "last_order_value": latest["order_value"],
                "last_freight_value": latest["freight_value"],
                "last_payment_installments": latest["payment_installments"],
                "last_payment_type": latest["payment_type"],
                "average_review_score": sum(valid_reviews) / len(valid_reviews) if valid_reviews else 0,
                "last_review_score": latest["review_score"] or 0,
                "average_delivery_delay_days": sum(delays) / len(delays) if delays else 0,
                "average_delivery_duration_days": sum(durations) / len(durations) if durations else 0,
            }
            snapshots.append(features)
            days_to_churn = INACTIVITY_DAYS - recency
            crosses_threshold = not any(
                order["date"] <= snapshot_date + timedelta(days=days_to_churn)
                for order in future_orders
            )
            labels.append(1 if crosses_threshold else 0)
            groups.append(customer_id)
        snapshot_date += timedelta(days=step_days)
    return snapshots, labels, groups


def extract_row_features(row):
    """Normalize row from synthetic 100k INR dataset or user upload into the 14 standard features."""
    shares = {
        "credit_card": float(row.get("credit_card_payment_share") or 0),
        "boleto": float(row.get("boleto_payment_share") or 0),
        "voucher": float(row.get("voucher_payment_share") or 0),
        "debit_card": float(row.get("debit_card_payment_share") or 0),
    }
    top_ptype = max(shares, key=shares.get) if any(shares.values()) else row.get("last_payment_type", "credit_card")
    avg_delivery = float(row.get("avg_delivery_days") or row.get("average_delivery_duration_days") or 7.0)
    late_rate = float(row.get("late_delivery_rate") or 0.0)
    delay_days = float(row.get("average_delivery_delay_days") or (late_rate * avg_delivery))

    return {
        "days_since_last_order": float(row.get("recency_days_at_cutoff") or row.get("days_since_last_order") or 0),
        "orders_in_history": float(row.get("orders_4y") or row.get("orders_in_history") or 1),
        "days_since_first_order": float(row.get("tenure_days_feature_window") or row.get("days_since_first_order") or 365),
        "average_order_value": float(row.get("avg_order_value") or row.get("average_order_value") or 0),
        "total_historical_spend": float(row.get("total_spend_4y") or row.get("total_historical_spend") or 0),
        "average_items_per_order": float(row.get("avg_items_per_order") or row.get("average_items_per_order") or 1),
        "last_order_value": float(row.get("median_order_value") or row.get("last_order_value") or row.get("avg_order_value") or 0),
        "last_freight_value": float(row.get("avg_freight_per_order") or row.get("last_freight_value") or 0),
        "last_payment_installments": float(row.get("avg_payment_installments") or row.get("last_payment_installments") or 1),
        "last_payment_type": top_ptype,
        "average_review_score": float(row.get("avg_review_rating") or row.get("average_review_score") or 4.0),
        "last_review_score": float(row.get("avg_review_rating") or row.get("last_review_score") or 4.0),
        "average_delivery_delay_days": delay_days,
        "average_delivery_duration_days": avg_delivery,
    }


def build_ecommerce_model():
    """Train the supervised customer churn model on the 100k synthetic INR e-commerce dataset."""
    if SYNTHETIC_CSV.is_file():
        print(f"Loading 100,000 synthetic INR customer cohort from {SYNTHETIC_CSV}...")
        features = []
        labels = []
        with open(SYNTHETIC_CSV, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                features.append(extract_row_features(row))
                labels.append(int(float(row.get("churn") or 0)))

        print(f"Loaded {len(features):,} customer rows (Retained: {len(labels)-sum(labels):,}, Churned: {sum(labels):,}).")
        train_features, test_features, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=RANDOM_SEED, stratify=labels
        )

        vec = DictVectorizer(sparse=False)
        x_train = vec.fit_transform(train_features)
        x_test = vec.transform(test_features)

        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)

        clf = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_SEED, C=1.0)
        clf.fit(x_train, y_train)

        probs = clf.predict_proba(x_test)[:, 1]
        threshold = 0.5
        preds = (probs >= threshold).astype(int)

        cm = confusion_matrix(y_test, preds)
        metrics = {
            "roc_auc": round(roc_auc_score(y_test, probs), 3),
            "f1": round(f1_score(y_test, preds, zero_division=0), 3),
            "precision": round(precision_score(y_test, preds, zero_division=0), 3),
            "recall": round(recall_score(y_test, preds, zero_division=0), 3),
        }
        print("100k INR Churn Model -> " + ", ".join(f"{name}: {value:.3f}" for name, value in metrics.items()))
        print(f"Confusion Matrix (Test 20,000): TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]}")

        bundle = {
            "vectorizer": vec,
            "scaler": scaler,
            "classifier": clf,
            "decision_threshold": threshold,
            "majority_baseline_f1": round(f1_score(y_test, [1] * len(y_test), zero_division=0), 3),
            "label_name": "No repeat order within the next 90 days",
            "horizon": "90-day customer churn forecast calibrated on 100,000 INR e-commerce records",
            "inactivity_days": INACTIVITY_DAYS,
            "prediction_window_days": 90,
            "dataset_end": "2026-11-15",
            "metrics": metrics,
            "confusion_matrix": {
                "tn": int(cm[0, 0]),
                "fp": int(cm[0, 1]),
                "fn": int(cm[1, 0]),
                "tp": int(cm[1, 1]),
                "total_test": len(y_test),
                "repeat_specificity": round(float(cm[0, 0] / (cm[0, 0] + cm[0, 1])) * 100, 1),
                "churn_recall": round(float(cm[1, 1] / (cm[1, 0] + cm[1, 1])) * 100, 1),
            },
            "evaluation_split": "Held-out 20% stratified test split from 100,000 INR e-commerce orders.",
            "positive_snapshot_rate": round(sum(labels) / len(labels), 4),
            "training_customers": len(train_features),
            "test_customers": len(test_features),
            "limitations": "Trained on synthetic 100,000 INR customer dataset with bimodal spend distributions (DMart / Marketplace retail proxy). Feature weights represent observational associations with 90-day churn.",
            "feature_definitions": {
                "days_since_last_order": "Days between customer order cutoff and most recent purchase.",
                "orders_in_history": "Total completed orders observed for the customer.",
                "days_since_first_order": "Customer relationship duration in days.",
                "average_order_value": "Mean customer transaction basket value (INR).",
                "average_review_score": "Mean star rating (1-5 scale) across order history.",
                "average_delivery_delay_days": "Mean days delivered past promised SLA date.",
            },
        }
        joblib.dump(bundle, MODEL_DIR / "ecommerce.joblib")
        print(f"Saved 100k churn model to {MODEL_DIR / 'ecommerce.joblib'}")
        return bundle

    else:
        print("Synthetic dataset not found; falling back to Olist raw CSVs...")
        customer_orders = load_customer_orders()
        dataset_end = max(order["date"] for rows in customer_orders.values() for order in rows)
        features, labels, groups = make_snapshots(customer_orders, dataset_end)
        train_customers, test_customers = train_test_split(
            sorted(set(groups)), test_size=0.2, random_state=RANDOM_SEED
        )
        train_set, test_set = set(train_customers), set(test_customers)
        fit_idx = [i for i, customer in enumerate(groups) if customer in train_set]
        test_idx = [i for i, customer in enumerate(groups) if customer in test_set]

        vec = DictVectorizer(sparse=False)
        x_train = vec.fit_transform([features[i] for i in fit_idx])
        x_test = vec.transform([features[i] for i in test_idx])
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)
        clf = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_SEED, C=1.0)
        clf.fit(x_train, [labels[i] for i in fit_idx])
        probs = clf.predict_proba(x_test)[:, 1]
        metrics = {
            "roc_auc": round(roc_auc_score([labels[i] for i in test_idx], probs), 3),
            "f1": round(f1_score([labels[i] for i in test_idx], (probs >= 0.5).astype(int), zero_division=0), 3),
            "precision": round(precision_score([labels[i] for i in test_idx], (probs >= 0.5).astype(int), zero_division=0), 3),
            "recall": round(recall_score([labels[i] for i in test_idx], (probs >= 0.5).astype(int), zero_division=0), 3),
        }
        bundle = {
            "vectorizer": vec, "scaler": scaler, "classifier": clf, "decision_threshold": 0.5,
            "metrics": metrics, "inactivity_days": INACTIVITY_DAYS,
            "training_customers": len(train_customers), "test_customers": len(test_customers),
            "label_name": "Risk of reaching 90 days without a repeat purchase",
            "horizon": "Active customer approaching the 90-day inactivity threshold",
            "limitations": "Olist historical baseline",
        }
        joblib.dump(bundle, MODEL_DIR / "ecommerce.joblib")
        return bundle


if __name__ == "__main__":
    build_ecommerce_model()
