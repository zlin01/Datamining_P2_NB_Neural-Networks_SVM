import os
import string
from collections import defaultdict, Counter
import math
import sys
import time

def preprocess(text):
    text = text.lower()  # Convert to lowercase
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    return text.split()

def load_data(directory):
    # dictionary to store preprocessed data
    # https://www.geeksforgeeks.org/defaultdict-in-python/
    data = defaultdict(list)
    labels = []
    # go thought all files
    # https: // www.geeksforgeeks.org / python - os - listdir - method /
    for label in os.listdir(directory):
        label_dir = os.path.join(directory, label)
        # https://www.geeksforgeeks.org/python-os-path-join-method/
        if os.path.isdir(label_dir):
            for doc_file in os.listdir(label_dir):
                doc_path = os.path.join(label_dir, doc_file)
                with open(doc_path, 'r', encoding='utf-8', errors='ignore') as file:
                    file_data = file.read()
                    words = preprocess(file_data)
                    data[label].extend(words)  # Add words to the category's list in data
                    labels.append(label)  # Keep track of the document's category
    return data, labels

def train_naive_bayes(train_data):
    # store all unique words
    word = set()
    # Dictionary for count word f
    word_counts = defaultdict(Counter)
    total_word_counts = defaultdict(int)

    # Count words and calculate probabilities for each class
    for label, words1 in train_data.items():
        total_word_counts[label] += len(words1)
        word_counts[label].update(words1)
        word.update(words1)

    return {
        'word': word,
        'word_counts': word_counts,
        'total_word_counts': total_word_counts,
        'total_count': sum(total_word_counts.values())
    }

# Function to classify a single document
# https://zhuanlan.zhihu.com/p/37575364
# https://community.alteryx.com/t5/Data-Science/Naive-Bayes-in-Python/ba-p/138424
def classify(document, model):
    # set up a impossible best class and best log probability
    best_class, best_log_prob = None, float('-inf')
    # Calculate probabilities for each class
    for i in model['total_word_counts'].keys():
        log_prob = math.log(model['total_word_counts'][i] / model['total_count'])
        # Add log probability of each word in the document
        for word in document:
            word_count = model['word_counts'][i][word] + 1  # Laplace smoothing
            log_prob += math.log(word_count / (model['total_word_counts'][i] + len(model['word'])))
        # Update best class if current probability is higher
        if log_prob > best_log_prob:
            best_class, best_log_prob = i, log_prob
    return best_class

def evaluate_model(test_dir, model):
    correct = 0
    total = 0
    total_tp = 0 # total true positives
    total_fn = 0 # total false negatives
    # Iterate over all test documents and classify
    for category in os.listdir(test_dir):
        category_dir = os.path.join(test_dir, category)
        if os.path.isdir(category_dir):
            for doc_file in os.listdir(category_dir):
                doc_path = os.path.join(category_dir, doc_file)
                with open(doc_path, 'r', encoding='utf-8', errors='ignore') as file:
                    content = file.read()
                    words = preprocess(content)
                    predicted = classify(words, model)
                    correct += (predicted == category)
                    total += 1
                    if predicted == category:
                        total_tp += 1
                    else:
                        total_fn += 1

    accuracy = correct / total
    misclassified = total - correct
    buttom_sum = (total_tp+total_fn)
    recall = total_tp/ buttom_sum if buttom_sum > 0 else 0
    return accuracy, misclassified, recall

def naive_bayes(train_dir, test_dir):
    start_time = time.time()
    train_data, _ = load_data(train_dir)
    model = train_naive_bayes(train_data)
    accuracy, misclassified, recall = evaluate_model(test_dir, model)
    end_time = time.time()

    print(f'Accuracy: {accuracy}')
    print(f'Misclassified: {misclassified}')
    print(f'Recall: {recall}')
    print(f'Running Time: {end_time - start_time} seconds')


def task7_limiter(file_path, limit):
    with open(file_path, 'r') as file:
        words = [line.split()[0] for line in file.readlines()]
    return set(words[:limit])

def task7_naive_bayes(train_data, limited_vocab):
    word_counts = defaultdict(Counter)
    total_word_counts = defaultdict(int)

    # Count words and calculate probabilities for each class
    for category, words in train_data.items():
        # Keep only words in limited_vocab
        filtered_words = [word for word in words if word in limited_vocab]
        total_word_counts[category] += len(filtered_words)
        word_counts[category].update(filtered_words)

    return {
        'word': limited_vocab,  # Use the limited vocabulary
        'word_counts': word_counts,
        'total_word_counts': total_word_counts,
        'total_count': sum(total_word_counts.values())
    }

if __name__ == '__main__':
    train_directory = sys.argv[1]  # Training directory path
    test_directory = sys.argv[2]   # Test directory path
    naive_bayes(train_directory, test_directory)
    print("\n")
    print("Task7 part d")
    bag_of_words_path = 'BagofWords.txt'
    for dictionary_size in [70000, 50000, 30000, 10000]:
        start_time = time.time()
        # Load and limit word in bag of word
        limited_vocab = task7_limiter(bag_of_words_path, dictionary_size)
        # Load training data
        train_data, _ = load_data(train_directory)
        # Train Naive Bayes with limited word
        model = task7_naive_bayes(train_data, limited_vocab)
        # Evaluate the Naive Bayes
        accuracy, misclassified, recall = evaluate_model(test_directory, model)
        end_time = time.time()
        # Print results for each dictionary size
        print(f"Dictionary Size: {dictionary_size}")
        print(f"Accuracy: {accuracy}")
        print(f"Number of Misclassified: {misclassified}")
        print(f"Recall: {recall}")
        print(f"Running Time: {end_time - start_time} seconds\n")
#https://sylvanassun.github.io/2017/12/20/2017-12-20-naive_bayes/