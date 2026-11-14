from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether
from reportlab.lib.colors import HexColor
from pathlib import Path

OUT = Path('output/pdf/RetainX_Project_Complete_Guide.pdf')
W, H = A4
NAVY, BLUE, TEAL = HexColor('#15243A'), HexColor('#2559A6'), HexColor('#15847A')
PALE, LIGHT, MUTED, DARK = HexColor('#EEF4FA'), HexColor('#F4F6F8'), HexColor('#536273'), HexColor('#17202B')
S = getSampleStyleSheet()
S.add(ParagraphStyle(name='Kicker', parent=S['Normal'], fontName='Helvetica-Bold', fontSize=10, leading=13, textColor=TEAL, alignment=1, spaceAfter=12))
S.add(ParagraphStyle(name='Cover2', parent=S['Title'], fontName='Helvetica-Bold', fontSize=30, leading=35, textColor=NAVY, alignment=1, spaceAfter=12))
S.add(ParagraphStyle(name='Sub', parent=S['Normal'], fontSize=13, leading=19, textColor=MUTED, alignment=1))
S.add(ParagraphStyle(name='H1x', parent=S['Heading1'], fontName='Helvetica-Bold', fontSize=20, leading=24, textColor=NAVY, spaceBefore=5, spaceAfter=11, keepWithNext=True))
S.add(ParagraphStyle(name='H2x', parent=S['Heading2'], fontName='Helvetica-Bold', fontSize=13, leading=17, textColor=BLUE, spaceBefore=11, spaceAfter=5, keepWithNext=True))
S.add(ParagraphStyle(name='H3x', parent=S['Heading3'], fontName='Helvetica-Bold', fontSize=10, leading=13, textColor=TEAL, spaceBefore=7, spaceAfter=4, keepWithNext=True))
S.add(ParagraphStyle(name='Bodyx', parent=S['BodyText'], fontName='Helvetica', fontSize=9, leading=12.5, textColor=DARK, spaceAfter=5))
S.add(ParagraphStyle(name='Smallx', parent=S['BodyText'], fontSize=7.5, leading=9, textColor=MUTED, spaceAfter=3))
S.add(ParagraphStyle(name='Callx', parent=S['BodyText'], fontName='Helvetica-Bold', fontSize=9, leading=13, textColor=NAVY, backColor=PALE, borderColor=BLUE, borderWidth=.6, borderPadding=7, spaceBefore=3, spaceAfter=8))
S.add(ParagraphStyle(name='Codex', parent=S['Code'], fontName='Courier', fontSize=8, leading=11, textColor=DARK, backColor=LIGHT, borderPadding=6, spaceBefore=2, spaceAfter=7))
S.add(ParagraphStyle(name='THx', parent=S['BodyText'], fontName='Helvetica-Bold', fontSize=7.5, leading=9, textColor=colors.white))
S.add(ParagraphStyle(name='TCx', parent=S['BodyText'], fontName='Helvetica', fontSize=7.2, leading=9, textColor=DARK))
story=[]
def P(t, st='Bodyx'): story.append(Paragraph(t, S[st]))
def H(t, n=1): P(t, {1:'H1x',2:'H2x',3:'H3x'}[n])
def B(t): P('&bull; '+t)
def C(t): P(t,'Callx')
def T(rows, widths, header=True):
    data=[[Paragraph(str(v), S['THx'] if header and i==0 else S['TCx']) for v in row] for i,row in enumerate(rows)]
    t=Table(data,colWidths=widths,repeatRows=1 if header else 0,hAlign='LEFT')
    cmds=[('VALIGN',(0,0),(-1,-1),'TOP'),('GRID',(0,0),(-1,-1),.3,HexColor('#CBD4DE')),('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5),('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4)]
    if header: cmds += [('BACKGROUND',(0,0),(-1,0),NAVY),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,LIGHT])]
    else: cmds += [('ROWBACKGROUNDS',(0,0),(-1,-1),[colors.white,LIGHT])]
    t.setStyle(TableStyle(cmds)); story.extend([t,Spacer(1,6)])

# Cover
story += [Spacer(1,38*mm),Paragraph('PROJECT STUDY GUIDE / VIVA PREPARATION',S['Kicker']),Paragraph('RetainX',S['Cover2']),Paragraph('Complete Project Guide',S['Cover2']),Spacer(1,4*mm),Paragraph('Olist customer inactivity, 90-day churn rule, machine learning, evidence, recommendations, and application flow',S['Sub']),Spacer(1,18*mm)]
C('Purpose: understand what the project does, how each part works, how the model score is calculated, what went wrong in earlier versions, and how to explain the current limits honestly.')
story += [Spacer(1,28*mm),Paragraph('Prepared from the current project source code, model artifacts, and Olist training pipeline. November 2026.',S['Smallx']),PageBreak()]

H('How to use this guide')
P('This is a learning guide, not a claim that the churn classifier is production-ready. Sections 1-6 explain the problem and mathematics. Sections 7-10 explain code and the web app. Sections 11-14 cover recommendations, earlier problems, presentation answers, and the glossary.')
C('Core distinction: the application marks a customer inactive at 90 days using a direct rule. The classifier estimates whether an active customer will reach that 90-day point before placing another order. The classifier is experimental and currently underperforms a majority-class baseline on F1.')
H('Quick project facts',2)
T([['Item','Current project meaning'],['Dataset','Olist Brazilian e-commerce public data; historical order window ends in 2018.'],['Main business rule','90 days since last order means inactive/churned for this project.'],['Model target','At an active historical snapshot, will the customer reach 90 days since the last order before ordering again?'],['Features','Recency, order count, tenure, spend, prior reviews, delivery history, last order details.'],['Model','DictVectorizer + StandardScaler + class-weighted Logistic Regression.'],['Saved churn metrics','ROC-AUC 0.738; F1 0.776; precision 0.998; recall 0.635.'],['Majority baseline','F1 0.997; positive snapshot rate about 99.4%. Model F1 is worse.'],['Review model','Separate Portuguese TF-IDF sentiment classifier; auxiliary, not the churn model.']], [38*mm,132*mm])

H('1. Project purpose and problem')
P('In subscription businesses, churn can be observed when a user cancels. In e-commerce, customers usually do not cancel an account; they simply stop purchasing. Olist has no official churn column, so RetainX defines inactivity using elapsed days since the customer\'s latest order.')
H('What RetainX is intended to show',2)
B('How many days a customer has been inactive, calculated from the last purchase date.')
B('Whether that customer has crossed the chosen 90-day inactivity threshold.')
B('What prior customer-history signals the classifier associates with crossing the threshold before a repeat purchase.')
B('Practical follow-up actions based on observed history rather than generic advice.')
H('Key terms',2)
T([['Term','Plain-language meaning'],['Customer churn','A customer stopping purchases. Since there is no cancellation event, churn must be defined operationally.'],['Inactivity threshold','Chosen number of days without purchase. This project uses 90.'],['Snapshot','Historical “as of” date. Features use information available on or before this date.'],['Prediction horizon','Future period used to determine a training label. Here: time until crossing the 90-day inactivity boundary, with complete follow-up.'],['Proxy label','Approximate project label, not official ground truth supplied by Olist.'],['Feature','Input variable used by the model, such as days since last purchase.'],['Target / label','Outcome the model is trained to estimate.']], [38*mm,132*mm])

H('2. Dataset and tables')
P('Olist is a Brazilian marketplace dataset containing related CSV tables. Orders, customers, items, payments, reviews, and delivery timestamps are joined using IDs.')
T([['CSV table','Fields used','Why it matters'],['olist_orders_dataset.csv','order_id, customer_id, status, purchase/delivery/estimated timestamps','Defines delivered orders and date history.'],['olist_customers_dataset.csv','customer_id, customer_unique_id','Maps order-specific IDs to a stable customer identity.'],['olist_order_items_dataset.csv','order_id, price, freight_value','Calculates basket value, freight, item count.'],['olist_order_payments_dataset.csv','order_id, payment_value, installments, type','Adds payment details.'],['olist_order_reviews_dataset.csv','order_id, review_score, review_comment_message','Provides prior ratings and text.']], [47*mm,64*mm,59*mm])
H('Why customer_unique_id matters',2)
P('An order has a customer_id that can differ across records for the same real buyer. The training code maps customer_id to customer_unique_id, then groups all orders under that stable ID. This permits repeat-purchase history to be calculated at customer level.')
H('Data limits',2)
B('The data represents Olist activity in 2016-2018, not current behavior.')
B('The dataset ends; no later order in the file does not prove the customer never returned.')
B('Only delivered orders enter the purchase-history model.')
B('Olist provides no official churn label or approved 90-day rule.')

H('3. The 90-day logic and label construction')
H('Operational status at prediction time',2)
P('For a known current or historical date t, compute:')
P('days_inactive = (t - last_order_date).days','Codex')
P('If days_inactive >= 90, the interface marks inactive_90_days. Below 90, it marks active_under_90_days. This is a direct rule, not a machine-learning prediction.')
H('Historical training target',2)
B('Find the latest delivered order on or before the snapshot date.')
B('Calculate recency = snapshot date minus prior order date.')
B('Keep only snapshots with recency below 90 days, so the customer is active at the snapshot.')
B('Look forward to determine whether the customer orders before reaching the 90-day boundary.')
B('Label 1 if they reach 90 days without another purchase; label 0 if they order first.')
B('Exclude snapshots too close to the end of Olist to know the complete outcome.')
C('Example: on a snapshot the customer is 65 days from their last order. There are 25 days until the 90-day boundary. If they buy again in those 25 days, label 0; otherwise label 1. The model sees only data from before or on that snapshot.')
H('No data leakage',2)
P('Data leakage occurs when model inputs contain information known only after the prediction date. Future orders create the label but are not model features. Prior review and delivery values are used only when tied to orders that had already happened by the snapshot.')

H('4. Features and meanings')
T([['Feature','Definition at snapshot','Interpretation'],['days_since_last_order','Days from last prior order to snapshot','Recency; larger means closer to 90 days.'],['orders_in_history','Delivered orders to date','Purchase frequency/history depth.'],['days_since_first_order','Snapshot minus first order date','Customer tenure.'],['average_order_value','Prior spend / prior order count','Typical basket value.'],['total_historical_spend','Sum of prior order values','Observed spend to date.'],['average_items_per_order','Mean items across prior orders','Typical basket size.'],['last_order_value / last_freight_value','Most recent order value/shipping','Latest purchase context.'],['last_payment_installments / type','Payment detail from latest order','Transaction context.'],['average_review_score / last_review_score','Prior star rating mean/latest rating','Prior expressed satisfaction proxy.'],['average_delivery_delay_days','Mean delivered date minus estimate','Prior lateness; can be negative if early.'],['average_delivery_duration_days','Mean delivery minus purchase date','Prior fulfillment speed.']], [48*mm,67*mm,55*mm])
P('A review score of 0 means no usable prior score was present, not zero stars. Date differences use whole calendar days in this code.')

H('5. Mathematics behind the model')
H('Step 1: dictionary to numeric columns',2)
P('DictVectorizer converts each feature dictionary into a fixed ordered vector. Numeric values stay numeric. Categories such as payment type become one-hot columns, for example last_payment_type=credit_card. Every row then has the same feature order.')
H('Step 2: standardize',2)
P('StandardScaler transforms each numeric column using training mean and standard deviation:')
P('z_j = (x_j - mu_j) / sigma_j','Codex')
P('x_j is a customer value, mu_j the training mean, and sigma_j the training standard deviation. The scaler is fit only on training rows, then applied to test rows.')
H('Step 3: Logistic Regression',2)
P('The model computes a linear score from standardized inputs:')
P('z_score = beta_0 + beta_1*z_1 + ... + beta_p*z_p','Codex')
P('The sigmoid maps it to a value between zero and one:')
P('p = 1 / (1 + exp(-z_score))','Codex')
P('Class 1 means the customer reaches the 90-day inactivity boundary before ordering again. This is a model score, not a calibrated probability of permanent churn.')
H('Class weights and regularization',2)
P('class_weight="balanced" increases the training penalty for errors on the rarer class. It does not create new examples or guarantee better precision/recall. Logistic Regression uses L2 regularization by default; C=1.0 controls inverse regularization strength. Smaller C penalizes large coefficients more.')
H('Decision threshold',2)
P('The classifier threshold is 0.5: predicted class 1 when p >= 0.5. This is separate from the operational 90-day status rule. Changing the model cutoff trades precision and recall; it does not alter the 90-day definition.')

H('6. Validation and interpretation')
P('The code splits unique customer IDs, then assigns every snapshot from a customer to the same partition. This prevents one customer\'s history from leaking across train/test. It is a random customer holdout, not a future calendar-time evaluation.')
P('Saved bundle: 193,121 snapshots from 70,418 unique customers; 56,334 train customers and 14,084 test customers. The full-follow-up cutoff is 2018-08-29. Positive snapshot rate is 99.4%.')
T([['Metric','Value','Meaning'],['ROC-AUC','0.738','Ranking across cutoffs; 0.5 is random. Does not establish calibration.'],['F1','0.776','Harmonic mean of precision and recall at chosen cutoff.'],['Precision','0.998','Among predicted positives, fraction truly positive.'],['Recall','0.635','Fraction of actual positives found.'],['Positive rate','99.4%','Share of snapshots that cross threshold.'],['Majority baseline F1','0.997','Always predicting the majority class; higher than the model.']], [39*mm,28*mm,103*mm])
C('Main limitation: the fitted classifier does not beat the majority-only F1 baseline. Its ROC-AUC suggests some ranking signal, but its current 0.5-threshold classification is not useful by F1. Do not call it accurate or production-ready. The 90-day rule is the dependable status assessment.')
H('Metric formulas',2)
B('True positive (TP): predicts crossing and customer crosses.')
B('False positive (FP): predicts crossing but customer orders before 90 days.')
B('False negative (FN): predicts an order first, but customer crosses threshold.')
P('Precision = TP/(TP+FP). Recall = TP/(TP+FN). F1 = 2*precision*recall/(precision+recall). ROC-AUC measures ranking over all cutoffs. Accuracy is correct predictions / all predictions and is misleading with 99.4% prevalence.')

H('7. Explaining a prediction')
P('The code calculates approximate per-feature Logistic Regression contributions. After transforming and scaling the customer row, it multiplies each standardized value by its learned coefficient:')
P('contribution_j = z_j * beta_j','Codex')
P('A positive contribution moves the linear score toward class 1; a negative contribution moves it away. The API sorts by absolute value and returns the top five. This explains the model score, not the cause of customer behavior. The intercept is omitted.')
C('“This customer is at 78 days since the last purchase, so they are 12 days below our 90-day rule. The model score is influenced most by days since last order and prior order count. These are associations learned from Olist, not proof of why this customer stopped buying.”')

H('8. Code walkthrough: training')
T([['Function / constant','Purpose'],['ROOT, DATA_DIR, MODEL_DIR','Paths to project root, datasets/olist, and models/.'],['INACTIVITY_DAYS=90','Operational project threshold.'],['PREDICTION_WINDOW_DAYS=90','Follow-up required to determine historical outcome.'],['parse_date(value)','Parses timestamp strings; returns None for missing/invalid.'],['load_customer_orders()','Reads CSVs, maps customer IDs, aggregates items/payments/reviews, keeps delivered orders, sorts per customer.'],['make_snapshots(...)','Creates 30-day snapshots; computes past-only features; labels from future; excludes incomplete follow-up.'],['build_ecommerce_model()','Splits by customer, vectorizes, scales, fits Logistic Regression, computes metrics/baseline, saves bundle.']], [52*mm,118*mm])
H('Important implementation details',2)
P('Snapshots are generated every 30 days across calendar dates for each active customer. A customer can contribute multiple snapshots. Grouped splitting prevents cross-split leakage, but customers with longer histories can contribute more rows. A future-calendar holdout would provide a stronger test of later behavior.')
H('Saved artifacts',2)
B('models/ecommerce.joblib: vectorizer, scaler, classifier, threshold, metrics, label/horizon text, split description, and limitations.')
B('models/reviews.joblib: separate review text model; not the churn classifier.')
B('joblib: Python serialization library used to save/load trained model objects.')

H('9. Review sentiment auxiliary model')
P('train_reviews.py keeps non-empty Portuguese comments with scores 1, 2, 4, or 5. Scores 1-2 are negative (0); scores 4-5 positive (1); 3-star and blank comments are excluded.')
T([['Term','Meaning'],['TF-IDF','Weights words by frequency in a comment and rarity across comments.'],['Unigram / bigram','One-word and two-word tokens.'],['TfidfVectorizer','Learns vocabulary/IDF from training comments, converts text to sparse numeric vectors.'],['GroupShuffleSplit','Order-grouped holdout so comments from one order stay together.'],['Saved sentiment metrics','ROC-AUC .974, F1 .952, precision .951, recall .952.']], [48*mm,122*mm])
P('These scores measure prediction of rating-derived labels on a holdout. They do not prove independently human-annotated sentiment, and this text model does not determine churn.')

H('10. Flask application function and route guide')
T([['Function / route','Role'],['load_env_file()','Loads local .env values only when not already in environment.'],['load_model(industry)','Loads models/<industry>.joblib if available.'],['error_response(message,status)','Creates standard JSON error response.'],['clean_number(...)','Parses numeric input and validates finite value, range, and integer rule.'],['normalize_customer(...)','Validates form fields and builds model feature dictionary.'],['analyze_review(text)','Validates review; returns sentiment score/band if model exists.'],['make_evidence(bundle,features)','Transforms row, multiplies scaled values by coefficients, returns top five.'],['model_case(industry,customer)','Runs classifier; creates score/status/evidence; retrieves guidance.'],['fallback_recommendation(case)','Creates what/why/how response without Gemini.'],['gemini_reply(...,mode)','Calls Gemini; recommendation and chat have separate system instructions.'],['GET /','Renders dashboard/model availability.'],['GET /healthz','Reports model artifact availability.'],['POST /api/predict','Validates JSON and returns prediction, status, evidence, metrics.'],['POST /api/recommendations','Requires consent; generates initial briefing or fallback.'],['POST /api/chat','Requires consent, validates question, passes recent chat history.'],['413 / 429 handlers','JSON errors for oversized requests and AI rate limits.']], [55*mm,115*mm])
H('Form input caveat',2)
P('The form requests inactivity days, prior order count, customer history length, average order value, average review score, and average delivery delay. static/retainx.js fills some additional model fields with simple defaults (for example last order value equals average order value). Real use should populate all features from actual customer history rather than defaults.')

H('11. Website, recommendations, and chatbot')
B('templates/index.html defines page structure, input fields, results, consent, and chat.')
B('static/retainx.js loads sample profiles, collects fields, calls API routes, renders evidence, and keeps recent chat history.')
B('app.py validates input, runs models, retrieves guidance, and calls Gemini.')
B('knowledge_base.py chooses advice through simple customer-history conditions.')
T([['Mode','Expected response'],['Initial recommendation','Short WHAT / WHY / HOW / WHAT TO DO briefing grounded in actual values and retrieved guidance.'],['Follow-up chat','Answer only newest question; follow “one line”, “three bullets”, or similar format request. History resolves references.'],['Preview mode','Deterministic fallback when Gemini key is absent; should still be tailored to current customer values.']], [40*mm,130*mm])
P('Prompt instructions improve response style but cannot guarantee perfect compliance from every LLM response. RAG means retrieved reference text is inserted into the prompt before generation. Here retrieval uses Python conditions, not an independently verified knowledge system.')
P('Consent gates recommendations/chat. Flask-Limiter allows 30 AI calls/hour per shared scope. MAX_CHAT_MESSAGE is 1,200 characters; request size is capped at 32 KB; review text at 2,500 characters.')

H('12. Problems faced and lessons')
T([['Problem','Why it mattered','Lesson / change'],['AI-generated code was not understood','The model could not be defended in viva.','Learn the target, features, math, and limitations in this guide.'],['Original target was “ever single-order customer”','Retrospective, not a prospective time-based churn prediction.','Moved to historical snapshots and 90-day inactivity.'],['Random order-level split','Same customers could appear in train and test.','Now split by customer ID.'],['Post-purchase inputs used as if predictive','Information might not exist at intervention time.','Features now use past-only snapshot history.'],['Few repeat orders within 90 days','Target is highly imbalanced.','Report the baseline; classifier F1 is below it.'],['Headline metrics misled','Majority prediction scored very high.','Do not call model useful based on accuracy/F1 alone.'],['Recommendations were technical/repetitive','Same long prompt used for all questions.','Initial briefing and follow-up chat now use different prompts.'],['Old RAG rules used removed form fields','Recommendations became generic.','Triggers now use current history fields.']], [42*mm,58*mm,70*mm])

H('13. Viva questions and clear answers')
qa=[('What is churn here?','We define an operational rule: 90 days without an order means inactive. Olist has no official churn column.'),('What does the model predict?','At an active historical snapshot, it estimates whether the customer reaches 90 days since the last order before buying again.'),('Why both rule and model?','The 90-day rule determines current inactive status; the classifier estimates risk before that threshold. They are different outputs.'),('Does a score of 70% mean a true 70% chance?','No. It is a model score and is not calibrated. Current performance is below the majority F1 baseline, so I would not call it a true real-world probability.'),('How do you avoid leakage?','Inputs use orders on or before a snapshot; future orders only create the target.'),('How is the split done?','Unique customer IDs are divided, and all snapshots for a customer stay in one side.'),('Why is the classifier not good yet?','About 99.4% of snapshots cross the threshold. Majority-only F1 is 0.997 versus model F1 0.776. ROC-AUC 0.738 suggests ranking signal, but not useful classification at this cutoff.'),('What do explanation values mean?','Standardized customer value times fitted coefficient. It explains model score movement, not causation.'),('How do reviews affect churn?','Prior star ratings are churn features. The text sentiment classifier is separate and predicts rating-derived labels.'),('What is TF-IDF?','Text representation that weights terms by use in a comment and rarity across comments.'),('What does Logistic Regression do?','Weighted sum of inputs followed by sigmoid to estimate a binary-class score.'),('What does chatbot do?','Initial recommendation covers what/why/how/actions. Follow-up answers the latest question and requested format.'),('What is honest conclusion?','The 90-day rule is transparent. The classifier is experimental and does not beat a simple F1 baseline.')]
for q,a in qa: story.append(KeepTogether([Paragraph(q,S['H3x']),Paragraph(a,S['Bodyx'])]))

H('14. Suggested presentation script')
P('“Our project uses Olist marketplace data to study non-contractual churn. We define inactivity as 90 days without a purchase. We create historical snapshots and use only history available by that date: recency, prior order count, tenure, spend, ratings, and delivery performance. The target records whether the customer reaches the 90-day inactivity boundary before buying again. A Logistic Regression model estimates that outcome, and the app shows coefficient-based evidence. We split by customer. However, about 99.4% of snapshots cross the threshold, so a majority baseline has F1 0.997 while our model gets 0.776. Therefore the explicit inactivity rule is the primary status and the classifier is experimental. Recommendations translate the evidence into plain-language actions, and follow-up chat respects the user’s requested format.”')

H('15. Glossary')
T([['Term','Definition'],['API','Application Programming Interface; endpoints browser calls.'],['Artifact','Saved fitted model and preprocessing objects.'],['Baseline','Simple reference such as always choosing majority class.'],['Class imbalance','One label appears much more often than another.'],['Coefficient (beta)','Learned weight affecting model linear score.'],['Data leakage','Using information unavailable at prediction time.'],['F1','Harmonic mean of precision and recall.'],['Feature engineering','Creating model inputs from raw tables.'],['Holdout','Data kept separate from model fitting for evaluation.'],['Label encoding','Represent target classes numerically (0/1).'],['One-hot encoding','Create a binary column for each category value.'],['Precision','Share of predicted positives that are actually positive.'],['Recall','Share of all actual positives detected by model.'],['ROC-AUC','Ranking performance across classification thresholds.'],['Standardization','Center/scale values with training mean and standard deviation.'],['Training','Fitting model parameters from labeled examples.'],['Validation/test','Data used to estimate generalization.']], [40*mm,130*mm])

H('16. How to run and retrain')
P('Install dependencies with pip install -r requirements.txt, place Olist files under datasets/olist, then run python train_models.py to rebuild churn and python train_reviews.py for sentiment. Start with python -m flask --app app run --port 5000. Run existing tests with python -m unittest discover -s tests -p "test_*.py".')
C('Before presenting: test customers at 30, 89, and 90 inactive days; explain the status change at 90 and compare model F1 with the majority baseline.')

def footer(canvas, doc):
    canvas.saveState()
    if doc.page > 1:
        canvas.setStrokeColor(HexColor('#D8E0E8')); canvas.setLineWidth(.5); canvas.line(20*mm,15*mm,W-20*mm,15*mm)
        canvas.setFont('Helvetica',7.5); canvas.setFillColor(MUTED); canvas.drawString(20*mm,10*mm,'RetainX | Complete Project Guide | November 2026'); canvas.drawRightString(W-20*mm,10*mm,f'Page {doc.page}')
    canvas.restoreState()

doc=SimpleDocTemplate(str(OUT),pagesize=A4,rightMargin=20*mm,leftMargin=20*mm,topMargin=18*mm,bottomMargin=21*mm,title='RetainX Complete Project Guide',author='RetainX Project Study Guide')
doc.build(story,onFirstPage=footer,onLaterPages=footer)
print(OUT.resolve(),OUT.stat().st_size)
