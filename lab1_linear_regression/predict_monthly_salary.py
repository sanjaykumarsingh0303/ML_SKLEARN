# Linear regression predicts a number. Here that number is monthly salary from years of experience.
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# One row is one employee. The feature is years of experience.
years_of_experience = np.array(
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 10],
    dtype=float,
).reshape(-1, 1)
monthly_salary_inr = np.array(
    [25000, 30000, 35000, 40000, 48000, 55000, 60000, 68000, 75000, 90000],
    dtype=float,
)

salary_model = LinearRegression()
salary_model.fit(years_of_experience, monthly_salary_inr)

# Each inner list is one employee. 2 and 5 are years of experience.
experience_to_predict = np.array([[2], [5]])
predicted_salaries = salary_model.predict(experience_to_predict)

print(f"Extra salary per year of experience: {salary_model.coef_[0]:,.0f} INR")
print(f"Starting salary at 0 years: {salary_model.intercept_:,.0f} INR")
print(f"Predicted salary for 2 years: {predicted_salaries[0]:,.0f} INR")
print(f"Predicted salary for 5 years: {predicted_salaries[1]:,.0f} INR")

plt.scatter(years_of_experience, monthly_salary_inr, color="blue", label="Known salaries")
plt.plot(
    years_of_experience,
    salary_model.predict(years_of_experience),
    color="red",
    label="Predicted line",
)
plt.scatter(experience_to_predict, predicted_salaries, color="green", label="New predictions")
plt.xlabel("Years of experience")
plt.ylabel("Monthly salary (INR)")
plt.title("Salary from years of experience")
plt.legend()
plt.show()
