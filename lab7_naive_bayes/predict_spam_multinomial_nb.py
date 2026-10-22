# MultinomialNB predicts spam or not spam from how many times each word appears.
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB

# Columns are word counts: free, win, hello.
# 1 means spam, 0 means not spam.
word_counts = np.array(
    [
        [4, 2, 0],
        [3, 3, 0],
        [5, 1, 1],
        [2, 4, 0],
        [0, 0, 3],
        [0, 1, 4],
        [1, 0, 2],
        [0, 0, 5],
    ],
    dtype=float,
)
is_spam = np.array([1, 1, 1, 1, 0, 0, 0, 0])

spam_model = MultinomialNB()
spam_model.fit(word_counts, is_spam)

# Two new emails, still counted as free, win, hello.
emails_to_predict = np.array(
    [
        [3, 2, 0],
        [0, 0, 2],
    ],
    dtype=float,
)
predicted_labels = spam_model.predict(emails_to_predict)
probability_of_spam = spam_model.predict_proba(emails_to_predict)[:, 1]

label_names = {0: "not spam", 1: "spam"}
print(f"free 3, win 2, hello 0: {label_names[predicted_labels[0]]}")
print(f"chance of spam: {probability_of_spam[0]:.0%}")
print(f"free 0, win 0, hello 2: {label_names[predicted_labels[1]]}")
print(f"chance of spam: {probability_of_spam[1]:.0%}")

plt.bar(
    ["free 3, win 2", "hello 2"],
    probability_of_spam,
    color=["red", "blue"],
)
plt.ylabel("Chance of spam")
plt.ylim(0, 1)
plt.title("MultinomialNB uses word counts")
plt.show()
