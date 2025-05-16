# FraudShield: Smart Credit Card Transaction Fraud Detection

## Overview
FraudShield is an intelligent machine learning project designed to detect fraudulent credit card transactions with high accuracy. Using a dataset of real European cardholders’ transactions, this project leverages multiple ensemble learning techniques to address the highly imbalanced classification problem of fraud detection.

## Dataset
The dataset contains transactions made over two days in September 2013, including 492 fraud cases out of 284,807 total transactions (~0.172% fraud). Features are anonymized principal components (`V1` to `V28`) derived by PCA, plus `Time` and `Amount`.

## Techniques Explored
To tackle the problem, I implemented and compared several ensemble learning algorithms:

- **Random Forest:** A bagging-based method leveraging multiple decision trees for robust classification.
- **Bagging:** Bootstrap aggregating with base estimators to reduce variance.
- **AdaBoost:** Boosting technique focusing on difficult-to-classify samples.
- **Gradient Boosting:** Sequential boosting to improve weak learners iteratively.
- **XGBoost:** An efficient and scalable implementation of gradient boosting.
- **Stacking:** Combining multiple base models with a logistic regression meta-learner for improved predictions.

## Model Selection and Results
After training and evaluation:

- **XGBoost achieved the highest testing accuracy (~99.96%),** outperforming other models in handling class imbalance and feature interactions.
- **Random Forest and Bagging** also performed excellently, with accuracies above 99.9%.
- **Stacking did not significantly improve accuracy** beyond the best single models but remains a powerful technique to consider with further tuning.
- **AdaBoost and Gradient Boosting** showed slightly lower accuracies, possibly due to the dataset’s high imbalance and noise.

## Why XGBoost?
- Handles imbalanced datasets well with built-in regularization.
- Efficient training and prediction speed.
- Robust performance on tabular data with complex feature interactions.
- Strong community support and continuous improvements.

## Additional Work
- Saved all trained models using `pickle` for future reuse.
- Built a **Streamlit-based interactive UI** to input transaction data and predict fraud in real-time.
- The UI accepts all 30 features (`Time`, `Amount`, `V1` to `V28`) and outputs a fraud probability with clear, user-friendly feedback.

## How to Use
1. Clone the repo.
2. Install requirements: `pip install -r requirements.txt`
3. Run the Streamlit app: `streamlit run streamlit_app.py`
4. Input transaction data manually or extend the app to support batch processing via CSV.

## Future Improvements
- Incorporate more explainability (SHAP values) to understand model decisions.
- Add automated threshold tuning to balance precision and recall based on use-case.
- Extend UI to support batch predictions with upload/download functionality.
- Experiment with anomaly detection models for unsupervised fraud detection.

---

*Developed by Tanish Rajput*

