import os
import sys
import time
import numpy as np
import tensorflow as tf
print(tf.__version__)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, recall_score

# Function to load text data and labels from a given directory
def load_data(directory):
    texts, labels = [], []
    # Iterate through each category folder
    for label in os.listdir(directory):
        label_dir = os.path.join(directory, label)
        # Iterate through each file in the category folder
        for filename in os.listdir(label_dir):
            filepath = os.path.join(label_dir, filename)
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as file:
                texts.append(file.read())  # Add file content to texts
                labels.append(label)  # Add category label to labels
    return texts, labels

# Function to preprocess text data
def preprocess_data(texts, labels, num_words):
    tokenizer = Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, padding='post')

    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)

    return padded_sequences, labels_encoded, len(label_encoder.classes_)

# Function to build a neural network model
# https://www.geeksforgeeks.org/training-of-convolutional-neural-network-cnn-in-tensorflow/?ref=ml_lbp
def build_model(feature_size, num_classes):
    model = Sequential([
        Embedding(feature_size, 10, input_length=None),  # Embedding layer 50->10
        GlobalAveragePooling1D(),  # Global Average Pooling
        Dense(4, activation='relu'),  # Hidden layer with 24 neurons ->4
        Dense(num_classes, activation='softmax')  # Output layer
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Main function to execute the training and testing of the neural network
def main(training_path, test_path, feature_size):

    start_time = time.time()

    # Load and preprocess the training and testing data
    train_texts, train_labels = load_data(training_path)
    test_texts, test_labels = load_data(test_path)

    train_data, train_labels, num_classes = preprocess_data(train_texts, train_labels, feature_size)
    test_data, test_labels, _ = preprocess_data(test_texts, test_labels, feature_size)

    # Build the model and train it
    model = build_model(feature_size, num_classes)
    model.fit(train_data, train_labels, epochs=10, verbose=1)

    # Predict and evaluate the model
    predictions = model.predict(test_data)
    predicted_labels = np.argmax(predictions, axis=1)

    accuracy = accuracy_score(test_labels, predicted_labels)
    misclassified = np.sum(predicted_labels != test_labels)
    recall = recall_score(test_labels, predicted_labels, average='macro')

    end_time = time.time()

    # Print the evaluation metrics
    print('Neural Networks')
    print(f'Feature Size: {feature_size}')
    print(f'Accuracy: {accuracy}')
    print(f'Misclassified: {misclassified}')
    print(f'Recall: {recall}')
    print(f'Running Time: {end_time - start_time:.2f} seconds\n')

if __name__ == '__main__':
    training_path = sys.argv[1]
    test_path = sys.argv[2]

    for feature_size in [30000, 10000]:
        main(training_path, test_path, feature_size)
