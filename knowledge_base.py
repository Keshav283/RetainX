"""
E-Commerce Retention Knowledge Base & Retrieval System (RAG)
Provides grounded, verified customer retention protocols tailored dynamically to specific e-commerce risk signals and K-Means behavioral segments.
"""

def retrieve_retention_protocols(customer_data, evidence=None, segment=None):
    """
    Dynamically select and customize verified e-commerce retention protocols
    based on the customer's actual telemetry, logistics, spending, review signals,
    and unsupervised K-Means customer segmentation profile.
    """
    retrieved = []
    days = customer_data.get("days_since_last_order", 0)
    threshold = customer_data.get("inactivity_days", 90)
    delay_days = customer_data.get("average_delivery_delay_days", 0)
    review_score = customer_data.get("average_review_score", 0) or customer_data.get("last_review_score", 0)
    order_val = customer_data.get("last_order_value", 0) or customer_data.get("average_order_value", 0)
    freight_val = customer_data.get("last_freight_value", 0)
    orders_count = customer_data.get("orders_in_history", 1)
    installments = customer_data.get("last_payment_installments", 1)

    # 1. Logistics & Delivery Delay Protocol
    if delay_days > 0:
        severity = "Severe" if delay_days >= 7 else "Moderate" if delay_days >= 3 else "Minor"
        freight_credit = max(50.0, round(freight_val, 2)) if freight_val else 100.0
        retrieved.append({
            "title": f"Logistics Service Recovery Protocol ({severity} Delay: +{delay_days:.1f} Days)",
            "condition": f"Carrier delivered package {delay_days:.1f} days past the promised estimated date.",
            "protocol": [
                f"Proactive Apology & Freight Credit: Automatically credit ₹{freight_credit:.2f} (freight reimbursement) or issue an immediate 20% discount coupon with personalized tracking apology.",
                "Carrier Performance Escalation: Flag regional delivery route and carrier partner for delivery SLA breach.",
                "VIP Fulfillment Priority: Tag customer profile for priority dispatch and automated SMS delivery checkpoints on subsequent orders."
            ]
        })
    elif delay_days < -2:
        retrieved.append({
            "title": "Early Delivery Delight Protocol (Delivered Ahead of Schedule)",
            "condition": f"Order arrived {abs(delay_days):.1f} days earlier than estimated delivery date.",
            "protocol": [
                "Positive Sentiment Check-in: Send a warm 'How was your delivery?' pulse message while the positive delivery impression is fresh.",
                "Referral Prompt: Provide a 'Share with a Friend' referral link offering ₹150 off for both the referee and the customer."
            ]
        })

    # 2. Negative Review & Sentiment Playbook
    if review_score and review_score <= 2:
        retrieved.append({
            "title": f"Dissatisfied Customer Service Recovery Playbook ({review_score}-Star Low Rating)",
            "condition": f"Customer gave an adverse rating of {review_score}/5 stars.",
            "protocol": [
                "Tier-2 Customer Success Outreach: Trigger immediate phone or personalized WhatsApp/email concierge outreach within 24 hours.",
                "Zero-Friction Resolution: Offer instant replacement or no-return-required store credit if items arrived broken, damaged, or incorrect.",
                "Merchant Review Escalation: Audit seller compliance and request seller remediation before seller rating drops below marketplace threshold."
            ]
        })
    elif review_score >= 4:
        retrieved.append({
            "title": f"Advocate Loyalty & Repeat Engagement Protocol ({review_score}-Star Highly Satisfied)",
            "condition": f"Customer rated their purchase highly ({review_score}/5 stars).",
            "protocol": [
                "Social Proof & Photo Review Reward: Invite customer to submit a photo review in exchange for loyalty bonus points.",
                "Category Affinity Recommendation: Suggest complementary accessories or replenishment items based on their past purchase."
            ]
        })

    # 3. Freight Cost / Basket Price Friction Protocol
    if order_val > 0 and freight_val > 0:
        freight_ratio = freight_val / (order_val + freight_val)
        if freight_ratio >= 0.20 or freight_val >= 35.0:
            retrieved.append({
                "title": f"High Shipping Cost Friction Mitigation (Freight: ₹{freight_val:.2f}, {freight_ratio:.1%} of basket)",
                "condition": f"Freight charge represents {freight_ratio:.1%} of total order expenditure.",
                "protocol": [
                    "Free Shipping Threshold Incentive: Issue a 'Free Shipping on Your Next Order Over ₹499' voucher valid for 30 days.",
                    "Regional Inventory Routing: Highlight local merchants situated in the customer's home state to reduce inter-state courier freight surcharges.",
                    "Bundle Suggestion: Educate customer on multi-item cart bundling to amortize fixed delivery costs across multiple items."
                ]
            })

    # 4. Inactivity & Lifecycle Win-back Protocols
    if days >= threshold:
        retrieved.append({
            "title": f"90-Day Lapsed Buyer Reactivation Campaign (Inactive for {days} Days)",
            "condition": f"Customer has crossed the {threshold}-day inactivity mark without a repeat order.",
            "protocol": [
                "Compelling Reactivation Offer: Send a high-impact 'We miss you' reactivation campaign with an exclusive ₹200 or 20% discount coupon.",
                "Catalog Update & New Arrivals: Showcase top trending items in their favorite category released since their last visit.",
                "Preference Pulse Survey: Ask a 1-click survey ('Was there a problem with your last order?') to diagnose permanent churn reasons."
            ]
        })
    elif days >= threshold * 0.65:
        remaining = threshold - days
        retrieved.append({
            "title": f"Imminent Inactivity Win-Back Cadence ({days} Days Inactive, {remaining} Days to 90-Day Churn)",
            "condition": f"Customer is approaching the {threshold}-day inactivity threshold ({remaining} days remaining).",
            "protocol": [
                "Time-Sensitive Reminder Offer: Trigger a personalized re-engagement email with an incentive expiring in 7 days.",
                "Cart Abandonment / Category Lifespan Follow-up: Cross-reference average product replenishment cycles and suggest restocking."
            ]
        })
    else:
        retrieved.append({
            "title": f"Active Customer Lifecycle Nurturing ({days} Days Since Last Order)",
            "condition": f"Customer is well within the active engagement window ({days} of {threshold} days).",
            "protocol": [
                "Loyalty Program Introduction: Inform customer of points accumulated and progress towards next VIP perk tier.",
                "Personalized Content & New Releases: Deliver editorial, educational, or seasonal recommendations tailored to their prior category."
            ]
        })

    # 5. Account Value & Installment Tier
    if order_val >= 500 or customer_data.get("total_historical_spend", 0) >= 2000:
        retrieved.append({
            "title": f"High-Value VIP Account White-Glove Playbook (Spend: ₹{order_val:.2f})",
            "condition": "Customer basket size or historical spend qualifies for VIP tier.",
            "protocol": [
                "Priority WhatsApp Support: Provide direct access to a dedicated retention concierge for fast query handling.",
                "Exclusive Flash Sale Previews: Grant 24-hour early access to upcoming marketplace promotional events.",
                "Custom Loyalty Milestone Gift: Add complimentary gift or double reward points on their subsequent checkout."
            ]
        })

    if installments >= 4:
        retrieved.append({
            "title": f"Financed Purchase Satisfaction Protocol ({installments} Installments)",
            "condition": f"Customer selected {installments} monthly payment installments.",
            "protocol": [
                "Post-Purchase Confidence Assurance: Reassure buyer of warranty and platform buyer-protection while monthly installment payments continue.",
                "Value Affirmation: Highlight positive peer reviews and longevity of purchased product to eliminate post-purchase buyer's remorse."
            ]
        })

    # 6. Unsupervised K-Means Behavioral Segment Protocols
    if segment and isinstance(segment, dict):
        seg_name = segment.get("name") or segment.get("short_name", "")
        if "VIP" in seg_name or "Champions" in seg_name:
            retrieved.append({
                "title": "VIP Champions White-Glove Retention Playbook (Cluster 3)",
                "condition": "Customer identified by K-Means as Top-Tier Enterprise Champion (High AOV & Spend).",
                "protocol": [
                    "Direct Account Manager: Assign dedicated VIP desk relationship manager with zero-hold WhatsApp support.",
                    "Zero-SLA Delivery Guarantee: All future orders shipped via express courier with waived freight fees.",
                    "Exclusive Member Preview: Early 24-hour access to high-demand brand drops and seasonal sales."
                ]
            })
        elif "Transit" in seg_name or "Delayed" in seg_name:
            retrieved.append({
                "title": "Logistics Friction Remediation Playbook (Cluster 1)",
                "condition": "Customer grouped in high delivery duration & delay cluster.",
                "protocol": [
                    "Automated Transit Refund: Issue ₹100 grocery voucher code with an apology for previous courier delays.",
                    "Fulfillment Re-assignment: Route subsequent packages via premier tier-1 national logistics partners.",
                    "Priority Packing Flag: Flag order at warehouse packing line for immediate same-day dispatch."
                ]
            })
        elif "At-Risk" in seg_name or "Lapsing" in seg_name:
            retrieved.append({
                "title": "At-Risk Lapsed Buyer Re-Activation Protocol (Cluster 2)",
                "condition": "Customer classified in 60+ days dormancy cluster.",
                "protocol": [
                    "Time-Sensitive Come-Back Credit: Issue ₹250 flat discount voucher valid for 7 days on minimum cart ₹999.",
                    "Restocking Notification: Send automated reminder for regularly purchased household categories.",
                    "Free Delivery Perk: Waive delivery fee on their next re-activation order."
                ]
            })
        elif "Active" in seg_name or "Regulars" in seg_name:
            retrieved.append({
                "title": "Active Regular Customer Loyalty Cadence (Cluster 0)",
                "condition": "Customer exhibits healthy repeat purchase frequency and low recency.",
                "protocol": [
                    "Tier Progression Milestone: Reward order with DMart Plus loyalty credits towards free grocery items.",
                    "Cross-Category Exploration: Introduce personalized recommendations in complementary categories.",
                    "Referral Bonus: Give ₹150 for every friend who places their first order."
                ]
            })

    return retrieved
