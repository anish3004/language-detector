#  Language Detector using Machine Learning

A machine learning project to detect the **language of a given text** from 22 different languages using the **Multinomial Naive Bayes** algorithm.

---

##  Dataset

- Source: `language.csv` containing 22,000 rows
- 22 languages with 1000 text samples each
- No missing values or null entries

---

##  Model and Tools Used

- **Python**
- **Scikit-learn** for machine learning
- **CountVectorizer** for feature extraction
- **Multinomial Naive Bayes** for classification
- **matplotlib** for visualization

---

##  How It Works

1. Text is transformed into numerical features using `CountVectorizer`.
2. Model is trained using `train_test_split` with 70% training data.
3. Accuracy of **~95.3%** is achieved on the test set.
4. User inputs a sentence → model predicts the language.

---

##  Confusion Matrix Insight

- Languages like **English**, **French**, **Spanish**, and **Urdu** show high accuracy.
- Misclassifications observed between **Latin, Portuguese, and Romanian** — likely due to linguistic similarity.

---

##  Demo Code

```python
user = input("Enter text: ")
data = cv.transform([user]).toarray()
output = model.predict(data)
print("Language detected to be:", output)
