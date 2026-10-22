# Linear regression predicts a number. Here that number is an exam score from hours studied.
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

hours_studied = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
exam_scores = np.array([35, 40, 50, 55, 60, 65, 70, 78, 85, 90], dtype=float)


def train_exam_score_model(features, target):
    # Keep 20% of the rows unseen so the score is not measured on data the model learned from.
    features_train, features_test, target_train, target_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=0,
    )

    exam_score_model = LinearRegression()
    exam_score_model.fit(features_train, target_train)
    return exam_score_model, features_test, target_test


def main():
    features = hours_studied.reshape(-1, 1)  # One column: hours studied for each student.
    target = exam_scores

    exam_score_model, features_test, target_test = train_exam_score_model(
        features,
        target,
    )

    points_per_extra_hour = exam_score_model.coef_[0]
    score_at_zero_hours = exam_score_model.intercept_
    predicted_scores = exam_score_model.predict(features_test)
    average_error = mean_absolute_error(target_test, predicted_scores)
    r2_score_value = r2_score(target_test, predicted_scores)

    hours_to_predict = np.array([[7.5]])  # One student who studied 7.5 hours.
    predicted_score = exam_score_model.predict(hours_to_predict)[0]

    print("points per extra hour:", round(points_per_extra_hour, 2))
    print("score at 0 hours:", round(score_at_zero_hours, 2))
    print("test predictions:", predicted_scores.round(1))
    print("average error:", round(average_error, 2))
    print("R2:", round(r2_score_value, 2))
    print("predicted score for 7.5 hours:", round(predicted_score, 1))


if __name__ == "__main__":
    main()
