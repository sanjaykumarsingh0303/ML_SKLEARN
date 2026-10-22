# KNN regression predicts salary as the average pay of the nearest employees.
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor

# One row is one employee. The feature is years of experience.
years_of_experience = np.array(
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 10],
    dtype=float,
).reshape(-1, 1)
monthly_salary_inr = np.array(
    [25000, 30000, 35000, 40000, 48000, 55000, 60000, 68000, 75000, 90000],
    dtype=float,
)

salary_knn = KNeighborsRegressor(n_neighbors=3)
salary_knn.fit(years_of_experience, monthly_salary_inr)

# Each inner list is one employee. 2 and 5 are years of experience.
experience_to_predict = np.array([[2], [5]])
predicted_salaries = salary_knn.predict(experience_to_predict)

print(f"Predicted salary for 2 years: {predicted_salaries[0]:,.0f} INR")
print(f"Predicted salary for 5 years: {predicted_salaries[1]:,.0f} INR")

experience_line = np.linspace(0, 10, 200).reshape(-1, 1)
plt.scatter(years_of_experience, monthly_salary_inr, color="blue", label="Known salaries")
plt.plot(
    experience_line,
    salary_knn.predict(experience_line),
    color="red",
    label="KNN prediction",
)
plt.scatter(experience_to_predict, predicted_salaries, color="green", label="New predictions")
plt.xlabel("Years of experience")
plt.ylabel("Monthly salary (INR)")
plt.title("Salary from 3 nearest employees")
plt.legend()
plt.show()
