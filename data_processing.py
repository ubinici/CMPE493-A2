import os
import re
import random
import string
import collections

# Load stopwords from a file
def load_stopwords(stopwords_path):
    with open(stopwords_path, 'r', encoding='utf-8') as f:
        stopwords = f.read().splitlines()
    return set(stopwords)

# Extract topics, title, and body from SGML content
def extract_data(content):
    topics = re.findall(r'<TOPICS>(.*?)</TOPICS>', content)
    topics = re.findall(r'<D>(.*?)</D>', topics[0]) if topics else []
    title = re.search(r'<TITLE>(.*?)</TITLE>', content)
    body = re.search(r'<BODY>(.*?)</BODY>', content)
    title = title[1] if title else ''
    body = body[1] if body else ''
    return topics, title, body

# Load data from SGML files in a directory
def load_data(data_path):
    data = []
    raw_data = []
    for filename in os.listdir(data_path):
        if filename.endswith(".sgm"):
            with open(os.path.join(data_path, filename), 'r', encoding='latin-1') as file:
                content = file.read()
                reuters_articles = content.split('</REUTERS>')
                for article in reuters_articles:
                    topics, title, body = extract_data(article)
                    # Filter out articles with no topics specified
                    if topics and (title or body):
                        data.append((topics, title, body))
                        raw_data.append(article)
    return raw_data, data

# Tokenize text, remove stopwords and punctuation
def tokenizer(text, stopwords):
    text = text.lower() # Convert text to lowercase
    text = re.sub(r'\d+', '', text) # Remove digits from the text
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text) # Remove punctuation from the text
    tokens = text.split()  # Split the text into tokens (words)
    tokens = list(filter(lambda token: token not in stopwords, tokens)) # Remove stopwords from the tokens
    return tokens

# Get the top N topics in the dataset
def get_top_n_topics(data, n):
    all_topics = [topic for topics, _ in data for topic in topics]
    counter = collections.Counter(all_topics)
    top_n_topics = [topic for topic, _ in counter.most_common(n)]

    multi_label_count = sum(
        len(set(topics) & set(top_n_topics)) > 1 for topics, _ in data
    )
    return top_n_topics, multi_label_count

# Preprocess data by tokenizing and removing stopwords
def preprocess_data(data, stopwords):
    preprocessed_data = []
    for topics, title, body in data:
        normalized_tokens = tokenizer(f'{title} {body}', stopwords)
        preprocessed_data.append((topics, normalized_tokens))
    return preprocessed_data

# Write preprocessed data to a file
def write_data_to_file(data, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        for topics, tokens in data:
            f.write(f'{topics}\n{tokens}\n\n')

# Get the total vocabulary size after preprocessing
def get_vocabulary_size(preprocessed_data):
    vocabulary = set()
    for _, tokens in preprocessed_data:
        vocabulary.update(tokens)
    return len(vocabulary)

# Get the number of documents labeled with more than one of the top 10 classes
def get_multi_label_count(data, top_n_topics):
    count = 0
    for topics, _ in data:
        intersection = [topic for topic in topics if topic in top_n_topics]
        if len(intersection) > 1:
            count += 1
    return count

# Split data into training, development, and test sets
def split_data(raw_data, preprocessed_data, train_ratio=0.8):
    train_data, dev_data, test_data = [], [], []
    for raw_article, (topics, tokens) in zip(raw_data, preprocessed_data):
        if lewissplit := re.search(r'LEWISSPLIT="(.*?)"', raw_article):
            lewissplit = lewissplit[1]
            if lewissplit == "TEST":
                test_data.append((topics, tokens))
            elif lewissplit == "TRAIN":
                train_data.append((topics, tokens))
    
    # Extract part of the training set as the development set
    train_size = int(len(train_data) * train_ratio)
    random.shuffle(train_data)
    dev_data = train_data[train_size:]
    train_data = train_data[:train_size]
    
    return train_data, dev_data, test_data
