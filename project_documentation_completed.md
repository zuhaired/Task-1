# Customer Churn Classification — Project Report

## 1. Objective
Explore customer churn data and compare Logistic Regression, Random Forest, and Gradient Boosting classifiers to assess whether customer characteristics can predict churn.

## 2. Dataset overview
- Rows: 5,880
- Columns: 21 (including `customerID` and target `Churn`)
- Target distribution: No = 2,982 (50.71%); Yes = 2,898 (49.29%)
- No pandas-null values were detected in the supplied CSV. `TotalCharges` is converted to numeric as a defensive preprocessing step.

## 3. Methodology
1. Load and inspect the CSV.
2. Drop `customerID`; encode `Churn` as Yes=1 and No=0.
3. Convert `TotalCharges` to numeric and handle any resulting missing values.
4. Split into stratified 80% training and 20% test sets.
5. One-hot encode categorical features and standardize numeric features in a scikit-learn pipeline.
6. Compare Logistic Regression, Random Forest, and Gradient Boosting using stratified five-fold cross-validation.
7. Evaluate held-out test predictions with accuracy, precision, recall, F1, ROC-AUC, confusion matrices, and ROC curves.

## 4. Verified test-set results
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.5009 | 0.4926 | 0.4000 | 0.4415 | 0.4748 |
| Random Forest | 0.5119 | 0.5054 | 0.4810 | 0.4929 | 0.4982 |
| Gradient Boosting | 0.4804 | 0.4721 | 0.4517 | 0.4617 | 0.4628 |

## 5. Interpretation and limitations
Random Forest has the highest accuracy and ROC-AUC among the three models in this run, but the ROC-AUC of 0.4982 is effectively random ranking. The other models also have ROC-AUC below 0.5. These results do **not** support a claim that the models can reliably predict churn. Do not deploy these predictions for customer-retention decisions.

Recommended next checks:
- Verify that each `Churn` label belongs to the correct customer row and that the label was constructed correctly.
- Confirm that features are not randomly generated or unrelated to the target.
- Inspect class labels and feature distributions, and compare against a simple baseline.
- Only report business associations after validating them directly against the dataset.

## 6. Running the project
Keep `Customer_Churn_Classification_Completed.ipynb`, `customer_churn_data.csv`, and `requirements_completed.txt` in the same directory. Install dependencies with `pip install -r requirements_completed.txt`, open the notebook in Jupyter, and run from top to bottom. The interactive prompt is not called automatically during batch execution; call `test_custom_prediction(best_pipeline)` in a notebook cell when you want to test a sample customer.

## 7. Conclusion
The notebook completes the exploration, preprocessing, model comparison, visualizations, and demonstration prediction workflow. The current dataset/model combination shows weak predictive signal, so the project conclusion is that further data and label validation is required before reliable churn prediction can be claimed.
