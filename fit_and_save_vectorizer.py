import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

# Load your data
df = pd.read_csv(r"C:\Users\Hp\Desktop\spam\spam.csv", encoding='ISO-8859-1')

# Rename columns
df.rename(columns={'v1': 'target', 'v2': 'text'}, inplace=True)

# Encode target labels
encoder = LabelEncoder()
df['target'] = encoder.fit_transform(df['target'])

# Visualize the data distribution
plt.pie(df['target'].value_counts(), labels=['ham', 'spam'], autopct="%0.2f")
# plt.show()

# Split the data into features and labels
X = df['text']
y = df['target']

# Initialize and fit the vectorizer
tfidf = TfidfVectorizer()
X_tfidf = tfidf.fit_transform(X)

# Initialize and train the model
model = MultinomialNB()
model.fit(X_tfidf, y)

# Save the fitted vectorizer
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)

# Save the trained model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
