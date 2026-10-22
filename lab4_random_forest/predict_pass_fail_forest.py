# Random forest classification predicts a label, here pass or fail.
# Many trees vote, and the majority vote is the prediction.
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

# One row is one student. 0 means fail, 1 means pass.
hours_studied = np.array(
    [1, 1.5, 2, 2.5, 3, 4, 5, 6, 7, 8],
    dtype=float,
).reshape(-1, 1)
exam_result = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

# n_estimators is the number of trees. max_depth keeps each tree small.
pass_fail_forest = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=0)
pass_fail_forest.fit(hours_studied, exam_result)

# Each inner list is one new student. The number is hours studied.
hours_to_predict = np.array([[2], [5]])
predicted_results = pass_fail_forest.predict(hours_to_predict)

result_names = {0: "fail", 1: "pass"}
print(f"2 hours studied: {result_names[predicted_results[0]]}")
print(f"5 hours studied: {result_names[predicted_results[1]]}")

# Blue points are known results. The red line is the forest vote. Green points are the new predictions.
hours_line = np.linspace(1, 8, 200).reshape(-1, 1)
plt.scatter(hours_studied, exam_result, color="blue", label="Known results")
plt.plot(
    hours_line,
    pass_fail_forest.predict(hours_line),
    color="red",
    label="Forest prediction",
)
plt.scatter(hours_to_predict, predicted_results, color="green", label="New predictions")
plt.xlabel("Hours studied")
plt.ylabel("0 = fail, 1 = pass")
plt.yticks([0, 1])
plt.title("Pass or fail from a random forest")
plt.legend()
plt.show()  # Close the chart window to let the program finish.
