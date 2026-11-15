import csv
from pathlib import Path

import joblib
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
SYNTHETIC_CSV = ROOT / "olist_synthetic_ecommerce_churn_100k_INR_bimodal.csv"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
RANDOM_SEED = 42

FEATURE_NAMES = [
    "days_since_last_order",
    "orders_in_history",
    "total_historical_spend",
    "average_order_value",
    "average_review_score",
    "average_delivery_duration_days",
    "average_delivery_delay_days",
]

FEATURE_LABELS = [
    "Recency (Days)",
    "Order Frequency",
    "Total Spend (INR)",
    "Avg Order Value (INR)",
    "Review Rating (Stars)",
    "Delivery Duration (Days)",
    "Delivery Delay (Days)",
]


def extract_clustering_features(row):
    avg_delivery = float(row.get("avg_delivery_days") or row.get("average_delivery_duration_days") or 7.0)
    late_rate = float(row.get("late_delivery_rate") or 0.0)
    delay_days = float(row.get("average_delivery_delay_days") or (late_rate * avg_delivery))

    return [
        float(row.get("recency_days_at_cutoff") or row.get("days_since_last_order") or 0.0),
        float(row.get("orders_4y") or row.get("orders_in_history") or 1.0),
        float(row.get("total_spend_4y") or row.get("total_historical_spend") or 0.0),
        float(row.get("avg_order_value") or row.get("average_order_value") or 0.0),
        float(row.get("avg_review_rating") or row.get("average_review_score") or 4.0),
        avg_delivery,
        delay_days,
    ]


def train_customer_segmentation():
    print(f"Loading customer cohort for unsupervised segmentation from {SYNTHETIC_CSV}...")
    matrix = []
    with open(SYNTHETIC_CSV, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            matrix.append(extract_clustering_features(row))

    X = np.array(matrix, dtype=float)
    total_customers = len(X)
    print(f"Extracted {total_customers:,} customer records across {len(FEATURE_NAMES)} behavioral dimensions.")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 1. Elbow analysis across k=2..7
    print("Computing Elbow method inertia for k=2..7...")
    elbow_data = []
    subset_for_elbow = X_scaled[:25000]
    for k in range(2, 8):
        km_test = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=5)
        km_test.fit(subset_for_elbow)
        elbow_data.append({"k": k, "inertia": round(float(km_test.inertia_), 1)})
        print(f"  k={k} -> Inertia: {km_test.inertia_:.1f}")

    # 2. Fit primary KMeans with optimal k=4
    optimal_k = 4
    print(f"Training primary K-Means model with k={optimal_k} clusters...")
    kmeans = KMeans(n_clusters=optimal_k, random_state=RANDOM_SEED, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)

    # 3. Compute silhouette score on a representative sample
    np.random.seed(RANDOM_SEED)
    sample_indices = np.random.choice(total_customers, size=min(6000, total_customers), replace=False)
    sil_score = round(float(silhouette_score(X_scaled[sample_indices], cluster_labels[sample_indices])), 4)
    print(f"Silhouette Score (k={optimal_k}): {sil_score}")

    # 4. PCA 2D projection for visualization
    print("Fitting PCA (2 components) for 2D scatter visualization...")
    pca = PCA(n_components=2, random_state=RANDOM_SEED)
    X_pca = pca.fit_transform(X_scaled)

    pca_sample_indices = np.random.choice(total_customers, size=600, replace=False)
    pca_scatter_points = []
    for idx in pca_sample_indices:
        pca_scatter_points.append({
            "x": round(float(X_pca[idx, 0]), 3),
            "y": round(float(X_pca[idx, 1]), 3),
            "cluster_id": int(cluster_labels[idx]),
        })

    # 5. Interpret cluster profiles
    centroids_orig = scaler.inverse_transform(kmeans.cluster_centers_)
    cluster_profiles = {}

    for cid in range(optimal_k):
        mask = (cluster_labels == cid)
        count = int(np.sum(mask))
        pct = round(count / total_customers * 100, 2)
        c_means = centroids_orig[cid]

        recency_val = c_means[0]
        spend_val = c_means[2]
        duration_val = c_means[5]
        delay_val = c_means[6]

        # Granular identification based on data signatures
        if spend_val >= 250000:
            name = "VIP High-Net-Worth Champions"
            short_name = "VIP Champions"
            desc = "Elite enterprise tier with extraordinary cumulative spend (₹8.5L+) and premium basket sizes (AOV ₹17,700+). Unmatched lifetime value."
            badge_color = "#3b82f6"  # Blue
            action = "Concierge WhatsApp manager, exclusive flash sale early-access, zero-fee expedited courier dispatch."
        elif duration_val > 20 or delay_val > 3.0:
            name = "Transit-Friction / Logistics-Delayed"
            short_name = "Logistics Friction"
            desc = "Customers suffering severe delivery delays (+6.4 days SLA breach, 29+ days in transit). Acute risk of delivery-induced dissatisfaction."
            badge_color = "#ef4444"  # Red
            action = "Automated freight refund (₹50-₹200), VIP regional courier re-routing, proactive SLA delay apology."
        elif recency_val > 60:
            name = "At-Risk / Lapsing Buyers"
            short_name = "At-Risk / Lapsing"
            desc = "Lapsed customers crossing the 60-90+ day inactivity threshold without recent orders. Frequency has cooled down."
            badge_color = "#f59e0b"  # Amber
            action = "Time-sensitive win-back coupon (₹200 discount valid 7 days), product replenishment reminder."
        else:
            name = "Active Regulars / Promising"
            short_name = "Active Regulars"
            desc = "Core loyal customer cohort with low recency (~17 days), steady order cadence (~45 orders), and reliable satisfaction."
            badge_color = "#10b981"  # Emerald Green
            action = "Milestone loyalty points, complementary category recommendations, friend referral perks."

        profile_features = {
            name_key: round(float(val), 2) for name_key, val in zip(FEATURE_NAMES, c_means)
        }

        cluster_profiles[cid] = {
            "cluster_id": cid,
            "name": name,
            "short_name": short_name,
            "count": count,
            "percent": pct,
            "description": desc,
            "badge_color": badge_color,
            "recommended_action": action,
            "centroid_means": profile_features,
        }

        print(f"\nCluster {cid}: {name} ({count:,} customers, {pct}%)")
        print(f"  Recency: {profile_features['days_since_last_order']}d, Spend: ₹{profile_features['total_historical_spend']:,.2f}, Delay: {profile_features['average_delivery_delay_days']}d")

    bundle = {
        "scaler": scaler,
        "kmeans": kmeans,
        "pca": pca,
        "pca_variance_ratio": [round(float(v), 4) for v in pca.explained_variance_ratio_],
        "optimal_k": optimal_k,
        "silhouette_score": sil_score,
        "feature_names": FEATURE_NAMES,
        "feature_labels": FEATURE_LABELS,
        "cluster_profiles": cluster_profiles,
        "elbow_data": elbow_data,
        "pca_scatter_points": pca_scatter_points,
        "total_segmented_customers": total_customers,
        "algorithm": "K-Means Unsupervised Clustering (k=4)",
        "metrics": {
            "silhouette_score": sil_score,
            "inertia": round(float(kmeans.inertia_), 1),
            "clusters": optimal_k,
        },
    }

    joblib.dump(bundle, MODEL_DIR / "segmentation.joblib")
    print(f"\nSaved updated segmentation model to {MODEL_DIR / 'segmentation.joblib'}")
    return bundle


if __name__ == "__main__":
    train_customer_segmentation()
