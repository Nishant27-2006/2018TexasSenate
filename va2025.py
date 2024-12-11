import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

data = {
    'text': [
        "Abigail Spanberger and Winsome Earle-Sears are tied at 39% support each.",
        "Poll shows Spanberger leading Miyares 40% to 39%.",
        "Youngkin's approval rating is at 46%, impacting the gubernatorial race.",
        "Voter sentiment is mixed with 43% feeling uncertain about Virginia's direction.",
        "Women favor Spanberger over Earle-Sears by a margin of 40% to 33%.",
        "Black voters show strong support for Spanberger at 62%."
    ],
    'label': ['Neutral', 'Neutral', 'Neutral', 'Neutral', 'Democrat', 'Democrat']
}

df = pd.DataFrame(data)

X = df['text']
y = df['label']

vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(report)
