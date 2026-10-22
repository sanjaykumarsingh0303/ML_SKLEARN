# Random forest regression predicts a number.
# It builds many decision trees on the same employees and averages their salaries.
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

# One row is one employee. reshape makes a single column of years of experience.
years_of_experience = np.array(
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 10],
    dtype=float,
).reshape(-1, 1)
monthly_salary_inr = np.array(
    [25000, 30000, 35000, 40000, 48000, 55000, 60000, 68000, 75000, 90000],
    dtype=float,
)

# n_estimators is the number of trees. max_depth stops each tree from memorizing every row.
salary_forest = RandomForestRegressor(n_estimators=100, max_depth=3, random_state=0)
salary_forest.fit(years_of_experience, monthly_salary_inr)

# Each inner list is one new employee. 2 and 5 are years of experience.
experience_to_predict = np.array([[2], [5]])
predicted_salaries = salary_forest.predict(experience_to_predict)

print(f"Predicted salary for 2 years: {predicted_salaries[0]:,.0f} INR")
print(f"Predicted salary for 5 years: {predicted_salaries[1]:,.0f} INR")

# Blue points are known salaries. The red line is the forest. Green points are the new predictions.
experience_line = np.linspace(0, 10, 200).reshape(-1, 1)
plt.scatter(years_of_experience, monthly_salary_inr, color="blue", label="Known salaries")
plt.plot(
    experience_line,
    salary_forest.predict(experience_line),
    color="red",
    label="Forest prediction",
)
plt.scatter(experience_to_predict, predicted_salaries, color="green", label="New predictions")
plt.xlabel("Years of experience")
plt.ylabel("Monthly salary (INR)")
plt.title("Salary from a random forest")
plt.legend()
plt.show()  # Close the chart window to let the program finish.
