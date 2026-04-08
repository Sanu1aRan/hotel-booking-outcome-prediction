Overview

Hotel Chain A was losing significant revenue due to booking cancellations and no-shows. This project builds a machine learning model to predict booking outcomes and help the hotel take early action.

Problem

Predict one of 3 booking outcomes from historical data:
•	Check-Out — guest completes their stay
•	Canceled — booking canceled before arrival
•	No-Show — guest never arrives without notice

Data

Provided directly by Hotel Chain A:
•	Training: 26,993 bookings after cleaning
•	Validation: 2,733 bookings
•	Test: 4,318 bookings
•	24 features including lead time, deposit type, room rate, guest history

Data Cleaning

•	Fixed inconsistent Reservation_Status labels
•	Removed negative lead times and zero length stays
•	Removed rows with missing Room Rate, Deposit Type and Visited Previously
•	Final training set: 26,993 rows



Exploratory Data Analysis

•	Analyzed cancellation patterns by lead time, deposit type and guest history
•	Visualized class imbalance — Check-Out 77%, Canceled 15%, No-Show 8%
•	Found lead time and deposit type as strongest predictors of cancellation
•	Identified regional and age group differences in cancellation patterns

Feature Engineering

•	Lead_Time — days between booking and check-in
•	Length_of_Stay — days between check-in and checkout
•	high_risk_flag — long lead time AND no deposit combined
•	family_size — total guests per booking
•	revenue_risk — room rate multiplied by lead time
•	Seasonality features — booking month, peak season, weekend flag

Approach

1. Label Encoding before SMOTE to avoid breaking synthetic samples
2. SMOTETomek to handle class imbalance — training data only
3. Compared 5 models using Macro F1 as primary metric
4. Selected best model and predicted on test set

Results

Best model: Decision Tree (Macro F1 = 0.348)

Macro F1 used instead of accuracy — dataset is imbalanced across 3 classes.

•	Logistic Regression: F1 = 0.302, Accuracy = 0.518
•	Decision Tree: F1 = 0.348, Accuracy = 0.453 (best)
•	Random Forest: F1 = 0.310, Accuracy = 0.543
•	XGBoost: F1 = 0.305, Accuracy = 0.547
•	ANN: F1 = 0.315, Accuracy = 0.517

Key Findings

•	1,594 out of 4,318 test bookings flagged as high risk (36.92%)
•	£1,985,459 revenue lost annually from cancellations and no-shows
•	Top features: Lead Time, Deposit Type, Ethnicity

Stack

Python, scikit-learn, XGBoost , imbalanced-learn , pandas , numpy , matplotlib , seaborn , Google Colab
