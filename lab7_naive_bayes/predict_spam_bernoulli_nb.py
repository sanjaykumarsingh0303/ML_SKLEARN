# BernoulliNB predicts spam or not spam from whether each word appears, not from the count.
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import BernoulliNB

# Columns are word presence: free, win, hello.
# 1 means the word appears. 0 means it does not.
# is_spam: 1 means spam, 0 means not spam.
word_present = np.array(
    [
        [1, 1, 0],
        [1, 1, 0],
        [1, 1, 1],
        [1, 1, 0],
        [0, 0, 1],
        [0, 1, 1],
        [1, 0, 1],
        [0, 0, 1],
    ],
    dtype=float,
)
is_spam = np.array([1, 1, 1, 1, 0, 0, 0, 0])

spam_model = BernoulliNB()
spam_model.fit(word_present, is_spam)

# Two new emails. 1 means the word appears.
emails_to_predict = np.array(
    [
        [1, 1, 0],
        [0, 0, 1],
    ],
    dtype=float,
)
predicted_labels = spam_model.predict(emails_to_predict)
probability_of_spam = spam_model.predict_proba(emails_to_predict)[:, 1]

label_names = {0: "not spam", 1: "spam"}
print("free yes, win yes, hello no:", label_names[predicted_labels[0]])
print(f"chance of spam: {probability_of_spam[0]:.0%}")
print("free no, win no, hello yes:", label_names[predicted_labels[1]])
print(f"chance of spam: {probability_of_spam[1]:.0%}")

plt.bar(
    ["free and win", "hello only"],
    probability_of_spam,
    color=["red", "blue"],
)
plt.ylabel("Chance of spam")
plt.ylim(0, 1)
plt.title("BernoulliNB uses word presence")
plt.show()
