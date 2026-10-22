# GaussianNB predicts a label from a measured number, here hours studied.
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB

# One row is one student. 0 means fail, 1 means pass.
hours_studied = np.array(
    [1, 1.5, 2, 2.5, 3, 4, 5, 6, 7, 8],
    dtype=float,
).reshape(-1, 1)
exam_result = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

pass_fail_nb = GaussianNB()
pass_fail_nb.fit(hours_studied, exam_result)

# Each inner list is one student. The number is hours studied.
hours_to_predict = np.array([[2], [5]])
predicted_results = pass_fail_nb.predict(hours_to_predict)
probability_of_pass = pass_fail_nb.predict_proba(hours_to_predict)[:, 1]

result_names = {0: "fail", 1: "pass"}
print(f"2 hours studied: {result_names[predicted_results[0]]}")
print(f"chance of passing after 2 hours: {probability_of_pass[0]:.0%}")
print(f"5 hours studied: {result_names[predicted_results[1]]}")
print(f"chance of passing after 5 hours: {probability_of_pass[1]:.0%}")

hours_line = np.linspace(1, 8, 200).reshape(-1, 1)
plt.scatter(hours_studied, exam_result, color="blue", label="Known results")
plt.plot(
    hours_line,
    pass_fail_nb.predict_proba(hours_line)[:, 1],
    color="red",
    label="Chance of passing",
)
plt.scatter(hours_to_predict, predicted_results, color="green", label="New predictions")
plt.xlabel("Hours studied")
plt.ylabel("0 = fail, 1 = pass")
plt.yticks([0, 1])
plt.title("Pass or fail with GaussianNB")
plt.legend()
plt.show()
