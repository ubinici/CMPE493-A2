import collections
import math

# Function to train a Bernoulli Naive Bayes classifier.
# Takes a list of training data and an optional alpha value for smoothing.
def train_bernoulli_nb(training_data, alpha=1.0):
    class_word_counts = collections.defaultdict(lambda: collections.defaultdict(int))
    class_counts = collections.defaultdict(int)

    # Iterate through the training data
    for topics, words in training_data:
        words_set = set(words)  # Create a set of unique words for Bernoulli NB
        for topic in topics:
            class_counts[topic] += 1
            for word in words_set:
                class_word_counts[topic][word] += 1

    # Calculate priors for each topic
    priors = {topic: count / len(training_data) for topic, count in class_counts.items()}
    
    # Create the vocabulary set
    vocabulary = set()
    for _, tokens in training_data:
        vocabulary.update(tokens)

    # Calculate likelihoods for each word and topic
    likelihoods = {topic: {word: (count + alpha) / (class_counts[topic] + 2 * alpha) for word, count in word_counts.items()} for topic, word_counts in class_word_counts.items()}

    return priors, likelihoods

# Function to classify a document using the trained Bernoulli Naive Bayes classifier.
# Takes a document, priors, and likelihoods as input.
def classify_bernoulli_nb(document, priors, likelihoods):
    scores = {}
    document_set = set(document)  # Create a set of unique words for Bernoulli NB
    
    # Calculate the score for each topic
    for topic, prior in priors.items():
        score = math.log(prior)
        for word in likelihoods[topic]:
            if word in document_set:
                score += math.log(likelihoods[topic][word])
            else:
                score += math.log(1 - likelihoods[topic][word])
        scores[topic] = score

    # Return the topic with the highest score
    return max(scores, key=scores.get)

# Function to evaluate the Bernoulli Naive Bayes classifier.
# Takes training data, development data, and an optional alpha value for smoothing.
def evaluate_bernoulli_nb(train_data, dev_data, alpha):
    priors, likelihoods = train_bernoulli_nb(train_data, alpha=alpha)
    tp, fp, fn = collections.defaultdict(int), collections.defaultdict(int), collections.defaultdict(int)

    # Iterate through the development data
    for topics, words in dev_data:
        predicted = classify_bernoulli_nb(words, priors, likelihoods)
        for topic in topics:
            if predicted == topic:
                tp[topic] += 1
            else:
                fn[topic] += 1
                fp[predicted] += 1

    # Calculate F-scores for each topic
    f_scores = {}
    for topic in tp.keys() | fp.keys() | fn.keys():
        precision = tp[topic] / (tp[topic] + fp[topic]) if tp[topic] + fp[topic] > 0 else 0
        recall = tp[topic] / (tp[topic] + fn[topic]) if tp[topic] + fn[topic] > 0 else 0
        f_scores[topic] = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

    # Calculate the micro and macro F-scores
    micro_f_score = sum(tp.values()) / (sum(tp.values()) + sum(fp.values()) + sum(fn.values()))
    macro_f_score = sum(f_scores.values()) / len(f_scores)

    return micro_f_score, macro_f_score

    # Function to test the Bernoulli Naive Bayes classifier.
    # Takes test data, priors, and likelihoods as input.
def test_bernoulli_nb(test_data, priors, likelihoods):
    correct = 0
        
    # Iterate through the test data
    for topics, words in test_data:
        predicted = classify_bernoulli_nb(words, priors, likelihoods)
        if predicted in topics:
            correct += 1
        
    # Calculate and return the accuracy
    return correct / len(test_data)

