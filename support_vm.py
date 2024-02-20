import os
import sys
import time
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, recall_score, classification_report
from sklearn.preprocessing import LabelEncoder

def load_data(directory):
    texts, labels = [], []
    for label in os.listdir(directory):
        label_dir = os.path.join(directory, label)
        for filename in os.listdir(label_dir):
            filepath = os.path.join(label_dir, filename)
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as file:
                texts.append(file.read())
                labels.append(label)
    return texts, labels

#https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
def preprocess_data(train_texts, test_texts, feature_size):
    # Create a Vectorizer Object
    vectorizer = CountVectorizer(max_features=feature_size, stop_words='english')
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)
    return X_train, X_test

def train_and_evaluate(X_train, y_train, X_test, y_test):
    # Train the model
    model = SVC(kernel='linear')
    model.fit(X_train, y_train)

    # Predict on test data
    predictions = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, predictions)
    misclassified = np.sum(predictions != y_test)
    recall = recall_score(y_test, predictions, average='macro')
    return accuracy, misclassified, recall

def main(training_path, test_path, feature_size):
    start_time = time.time()

    # Load and preprocess data
    train_texts, train_labels = load_data(training_path)
    test_texts, test_labels = load_data(test_path)
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_labels)
    y_test = label_encoder.transform(test_labels)

    X_train, X_test = preprocess_data(train_texts, test_texts, feature_size)

    # Train the model and evaluate
    accuracy, misclassified, recall = train_and_evaluate(X_train, y_train, X_test, y_test)

    end_time = time.time()

    print("SVM")
    print(f'Feature Size: {feature_size}')
    print(f'Accuracy: {accuracy}')
    print(f'Misclassified: {misclassified}')
    print(f'Recall: {recall}')
    print(f'Running Time: {end_time - start_time:.2f} seconds\n')
if __name__ == '__main__':
    training_path = sys.argv[1]
    test_path = sys.argv[2]
    for feature_size in [50000, 10000]:
        main(training_path, test_path, feature_size)
