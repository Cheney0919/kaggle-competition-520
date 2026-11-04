# AIPI 520 Kaggle Competition — Classical Music Meets Classical ML (Fall 2026)

Task: predict which previous patrons will buy a 2014–15 season subscription (AUROC).

Final model: logistic regression (public AUROC 0.9768).

## Run

1. Place competition data under `data/for_students/` (not included).
2. Run in order: `scripts/build_strict_features.py`, `scripts/build_clean_features.py`, `scripts/run_stage8.py`, `scripts/run_stage9.py`, `scripts/run_stage10.py`.

Requirements: Python, pandas, scikit-learn, xgboost.

## Not included

Raw competition data, derived feature tables, submission file, written report.


## Kaggle Competition Write-up: Predicting 2014–2015 Season Subscriptions
### Part 1. Data Understanding and Exploration
Kaggle Competition Write-up: Predicting 2014–2015 Season Subscriptions
Part 1. Data Understanding and Exploration
The goal of this competition is using their past behavior to predict the probability that an existing customer would subscribe to the 2014–2015 season‘s concert. But the training labels were binary, while we needed to predict probabilities. So this was still a binary classification problem. The evaluation metric was AUROC. It measures how well the model ranks subscribers above non-subscribers. The competition will use both public and private test scores, and the final score will base on their average. Only the public score is visible during submission, so I didn't want to keep adjusting my model just to improve the public leaderboard, which will make the private test maybe even worse.

There were 9 files, including 8 data tables and one sample submission. The training set has 6941 customers and the test set has 2975. The account table contains 19833 customers. The other tables include 2808 ticket records from 2049 customers, 28627 subscription records from 6267 customers, 103 concerts, 6 concerts for the target season, and 42522 ZIP code records. The sample submission has 2975 rows with 0.5 placeholder probabilities.

I noticed the class imbalance when I looked through data. Only about 4.8% of training customers subscribed to the target season. If I predicted everyone as a non-subscriber, I could still get around 95.2% accuracy. But that would not be a useful model. So I focused on AUROC instead of accuracy.

I also checked how tables connected to each other. Account ID is the main key. Tickets and Subscriptions had one-to-many relationships, because one customer can have multiple transactions. Concerts can be joined with Tickets by matching Season and Set. ZIP code information could be linked through Billing ZIP, but at last I didn't use the demographic data in my model.

The data was generally usable, but I still found some missing values and formatting issues. Shipping missed 98.7% data, and Relationship missed 96.7% data. There are 225 people with values in the relationship’s train set; the two largest categories are Education Participant (135 people) and Vendor - Artistic (74 people), with the rest being single digits. Relationship is totally another segment and was missing 96.7% data. This segment was different from regular patrons and didn't subscribe. All of their subscription labels are 0 aligned with that. So I just deleted this feature. Billing City and Billing ZIP missed 11.2% and 14.9%, so I kept them and handled the missing categories later. I also found BOM and encoding issues, different ZIP code format, and structural missing values. For example, 14298 accounts had no First Donated date, which matched accounts without donation records.

Another issue was possible data leakage. The dataset had two lifetime donation fields, amount.donated.lifetime and no.donations.lifetime, but their cutoff dates were not clearly defined. So I checked 126 customers whose first donation was in 2014. Their 2013 donation amounts were all zero, but their lifetime donation amounts were positive. This means that the lifetime amount included at least some 2014 information. The latest first donation date was September 11, 2014, but I couldn't confirm the exact data snapshot date.

Thus I decided to remove both lifetime donation fields. I kept the clearly defined 2013 donation amount and treated customers whose first donation was in 2014 as having no donation before 2014. I also checked the transaction tables and found no direct target-season subscription records. I still applied historical time restrictions when building features.

### Part 2. Feature Engineering
I joined the tables by using their corresponding keys and aggregated Tickets and Subscriptions by Account ID. I wanted the final dataset to have one row per customer instead of multiple rows.

19 features are selected: 6 account features, 6 ticket features, and 7 subscription features. After one-hot encoding, I had 174 columns, including 159 one-hot columns and 15 numerical columns.

I also reduced the number of City and ZIP Code categories. Some cities only had one or two records. I didn't think that keeping all these rare and tiny categories would help the model learn reliable patterns. So I kept 52 cities that appeared at least 20 times in the training set. Rare and unseen cities I turned them into Other, while missing values be Unknowns. 

I did something similar to ZIP Code. I standardized numeric ZIP codes to five digits, including ZIP+4 formats and shorter codes. Nonstandard or alphabetic codes were treated as Unknown. I kept 76 major ZIP codes, and converged the remaining rare or unseen valid codes into Other. 

I handled structural missing values separately. Missing Last Season values means no corresponding ticket or subscription history, so I used None to represent. For donation-related features, I only used information before 2014. All category thresholds and learned preprocessing rules were fitted only on training data.

### Part 3. Validation and Model Comparison
I used Stratified 5-Fold Cross-Validation instead of a single train-validation split. My dataset was relatively small and only about 4.8% of the samples were positive. I wanted every customer to be used once for validation, so I chose cross validation rather than regular split. Also five folds seemed like a reasonable balance between computation time and evaluation stability. I also used stratification to keep the class ratio similar across folds, and set the random seed as 42.

For Logistic Regression, I used One-Hot Encoding for categorical features and StandardScaler for numerical features. The scaler was fitted within each training fold to avoid leakage. StandardScaler was useful because the numerical features had different scales, and Logistic Regression uses regularization.

I used Logistic Regression as my baseline. It achieved a pooled OOF AUROC of 0.9623, with a mean fold AUROC of 0.9624 ± 0.0100.

I also tried XGBoost. I expected it to perform better than logistic regression because tree-based models are usually strong on tabular data and can capture nonlinear relationships. But the default XGBoost only achieved 0.9517. I tried tuning some parameters. Setting reg_alpha=1 gave 0.9595, reg_lambda=10 gave 0.9592, and max_depth=3 gave 0.9589. Even if I combined these settings, the score reduced to 0.9503. 

I also did an ablation study to see which feature groups were useful. I removed subscription features and XGBoost AUROC reduced from 0.9517 to 0.9022. This was a larger drop than removing geography or ticket features, which only reduced AUROC with 0.0016 and 0.0026. Removing donation features even improved AUROC by 0.0016. I also tested a very simple model using only ever_subscribed and donor_before_2014. Its AUROC was 0.8438, so those two features alone were not enough.

I think Logistic Regression worked well because many strong signals in this dataset were relatively simple. Customers who already subscribed before had a 13.6% subscription rate, and those who had never subscribed only had a 0.7% subscription rate. Customers who donated before 2014 also had a higher subscription rate, which was 13.5% vs. 1.5%. I think the reason why LR can perform better is that these patterns may work well with the additive log-odds structure of Logistic Regression.

The training set also had only about 332 positive samples. Complex tree models might learn noise from such limited data. Because the differences were comparable to the variation across folds, so I can't say this was definitely the reason, or that Logistic Regression was statistically better.

I tried an ensemble as well. It improved OOF AUROC from 0.9623 to 0.9632, only 0.0009 higher. I didn't think this small improvement was worth adding more complexity, especially without evidence to show that it would be stable.

I ended up choosing Logistic Regression as my final model. It was simple, relatively stable, and easier to explain. I trained it again by using the full training set and generated predictions for the test set.

### Part 4. Results and Limitations
I generated submission_lr_v1.csv with 2975 predictions. The ID order matched the sample submission, there were no missing values, and all predicted probabilities were between 0 and 1. The average predicted probability was 0.0485, it was close to the training positive rate which is 0.0478. 138 customers were predicted as subscribers at a 0.5 threshold.

My submission received a Public AUROC of 0.9768. This was higher than my OOF AUROC, but the private score was still unknown.

There were still some limitations. Most of my features came from historical subscriptions, tickets, donations, and account information. But real subscription decisions can also be affected by things like moving, health, income, or changes in personal interests. I couldn't observe these factors in the dataset. They might explain some of the remaining prediction errors.

I had also considered a Plan B. I wanted to build more detailed features from subscription history and match customers' composer preferences of the 2014–2015 concert program. But composer preference data only covered about 7.7% of the test customers. I wasn't sure the extra work would lead to a meaningful improvement. The practical significance might not be satisfied.

I could try more detailed subscription features, interaction terms, ZIP code demographic data, or CatBoost. Reliable alternative data might also help capture customer behavior that wasn't included in the original dataset.

I ended up keeping a relatively simple, stable, and interpretable model. I didn't want to keep increasing model complexity just to chase a small improvement that might not be worth the extra effort.


