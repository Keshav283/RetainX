import unittest
import app as retainx

class BatchE2ETests(unittest.TestCase):
    def setUp(self):
        self.client = retainx.app.test_client()

    def test_home_page_rendered(self):
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        self.assertIn(b"Batch Predict", res.data)
        self.assertIn(b"Unsupervised K-Means Clusters", res.data)

    def test_individual_prediction_with_segment(self):
        res = self.client.post("/api/predict", json={
            "industry": "ecommerce",
            "customer": {
                "days_since_last_order": 20,
                "orders_in_history": 5,
                "days_since_first_order": 180,
                "average_order_value": 1500,
                "total_historical_spend": 7500,
                "average_items_per_order": 2,
                "last_order_value": 1500,
                "last_freight_value": 50,
                "last_payment_installments": 1,
                "last_payment_type": "credit_card",
                "average_review_score": 5.0,
                "last_review_score": 5.0,
                "average_delivery_delay_days": 0,
                "average_delivery_duration_days": 5,
                "review_text": "Great delivery!"
            }
        })
        self.assertEqual(res.status_code, 200)
        body = res.get_json()
        self.assertIn("risk_percent", body)
        self.assertIn("customer_segment", body)
        self.assertIsNotNone(body["customer_segment"])
        self.assertIn("retention_protocols", body)

    def test_batch_predict_dmart_5k(self):
        with open("datasets/dmart_sample_30days_5k.csv", "rb") as f:
            res = self.client.post(
                "/api/batch-predict",
                data={"file": (f, "dmart_sample.csv")},
                content_type="multipart/form-data"
            )
        self.assertEqual(res.status_code, 200)
        body = res.get_json()
        self.assertEqual(body["total_customers"], 5000)
        self.assertEqual(body["summary"]["will_retain"] + body["summary"]["will_churn"], 5000)
        self.assertEqual(len(body["top_at_risk_customers"]), 50)
        self.assertEqual(len(body["segment_distribution"]), 4)

    def test_segmentation_stats(self):
        res = self.client.get("/api/segmentation-stats")
        self.assertEqual(res.status_code, 200)
        body = res.get_json()
        self.assertEqual(body["optimal_k"], 4)
        self.assertGreater(body["silhouette_score"], 0.2)

if __name__ == "__main__":
    unittest.main()
