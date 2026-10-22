# Logistic regression predicts a label. Here the label is malignant or benign.
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# 0 is malignant and 1 is benign. The features are measurements from each tumor.
features, diagnosis = load_breast_cancer(return_X_y=True)

# Train on 80% of the rows. Score the model on the other 20%.
features_train, features_test, diagnosis_train, diagnosis_test = train_test_split(
    features,
    diagnosis,
    test_size=0.20,
    random_state=23,
)

# max_iter gives the solver enough steps to finish on this dataset.
cancer_model = LogisticRegression(max_iter=10000, random_state=0)
cancer_model.fit(features_train, diagnosis_train)

predicted_diagnosis = cancer_model.predict(features_test)
accuracy_percent = accuracy_score(diagnosis_test, predicted_diagnosis) * 100
print(f"Logistic Regression model accuracy: {accuracy_percent:.2f}%")
