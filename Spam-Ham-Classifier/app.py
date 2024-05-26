import pandas as pd
import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Download necessary NLTK data
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

# Load the data
data = pd.read_csv('spam_ham.csv')

# Check dataset balance
print(data['label'].value_counts())

# Preprocess the text
def preprocess_text(text):
    # Tokenization
    tokens = word_tokenize(text.lower())
    
    # Stopwords removal
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    # Remove unwanted symbols (keep only alphabetic characters)
    tokens = [re.sub(r'[^a-zA-Z0-9]', '', word) for word in tokens if re.sub(r'[^a-zA-Z0-9]', '', word) != '']
    
    return ' '.join(tokens)

data['processed_text'] = data['text'].apply(preprocess_text)

# Convert text to numerical features using TF-IDF
tfidf_vectorizer = TfidfVectorizer()
X = tfidf_vectorizer.fit_transform(data['processed_text'])
y = data['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train a classifier
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

# Predict and evaluate the classifier
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')
print('Classification Report:')
print(classification_report(y_test, y_pred, zero_division=1))