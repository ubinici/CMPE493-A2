import math

# Function to train a Multinomial Naive Bayes classifier.
# Takes a list of training data and an optional alpha value for smoothing.
def train_multinomial_nb(training_data, alpha=1.0):
    # Initialize the count dictionaries and the vocabulary set
    class_counts = {}
    class_word_counts = {}
    vocab = set()

    # Iterate through the training data
    for topics, words in training_data:
        for topic in topics:
            # Update class_counts dictionary
            class_counts[topic] = class_counts.get(topic, 0) + 1

            # Update class_word_counts dictionary
            class_word_counts.setdefault(topic, {})
            for word in words:
                vocab.add(word)
                class_word_counts[topic][word] = class_word_counts[topic].get(word, 0) + 1

    # Calculate priors for each topic
    priors = {topic: count / len(training_data) for topic, count in class_counts.items()}

    # Calculate likelihoods for each word and topic
    likelihoods = {}
    for topic in class_word_counts:
        likelihoods[topic] = {}
        total_words = sum(class_word_counts[topic].values())
        for word in vocab:
            likelihoods[topic][word] = (class_word_counts[topic].get(word, 0) + alpha) / (total_words + alpha * len(vocab))

    return priors, likelihoods

# Function to classify a document using the trained Multinomial Naive Bayes classifier.
# Takes a document, priors, and likelihoods as input.
def classify_multinomial_nb(document, priors, likelihoods):
    scores = {}
    
    # Calculate the score for each topic
    for topic, prior in priors.items():
        score = math.log(prior)
        for word in document:
            if word in likelihoods[topic]:
                score += math.log(likelihoods[topic][word])
        scores[topic] = score

    # Return the topic with the highest score
    return max(scores, key=scores.get)

# Function to evaluate the Multinomial Naive Bayes classifier.
# Takes training data, development data, and an optional alpha value for smoothing.
def evaluate_multinomial_nb(train_data, dev_data, alpha):
    priors, likelihoods = train_multinomial_nb(train_data, alpha=alpha)
    tp, fp, fn = {}, {}, {}

    # Iterate through the development data
    for topics, words in dev_data:
        predicted = classify_multinomial_nb(words, priors, likelihoods)
        for topic in topics:
            if predicted == topic:
                tp[topic] = tp.get(topic, 0) + 1
            else:
                fn[topic] = fn.get(topic, 0) + 1
                fp[predicted] = fp.get(predicted, 0) + 1

    # Calculate F-scores for each topic
    f_scores = {}
    for topic in set(tp) | set(fp) | set(fn):
        precision = (tp.get(topic, 0) + alpha) / (tp.get(topic, 0) + fp.get(topic, 0) + alpha)
        recall = (tp.get(topic, 0) + alpha) / (tp.get(topic, 0) + fn.get(topic, 0) + alpha)
        f_scores[topic] = 2 * precision * recall / (precision + recall)

    # Calculate the macro and micro F-scores
    macro_f_score = sum(tp.values()) / (sum(tp.values()) + sum(fp.values()))
    # Calculate the macro and micro F-scores
    macro_f_score = sum(tp.values()) / (sum(tp.values()) + sum(fp.values()) + sum(fn.values()))
    micro_f_score = sum(f_scores.values()) / len(f_scores)

    return micro_f_score, macro_f_score


# Function to test the performance of the Multinomial Naive Bayes classifier.
# Takes test data, priors, and likelihoods as input.
def test_multinomial_nb(test_data, priors, likelihoods):
    correct = 0

    # Iterate through the test data
    for topics, words in test_data:
        predicted = classify_multinomial_nb(words, priors, likelihoods)

        # Check if the predicted topic is in the true topics
        if predicted in topics:
            correct += 1

    # Calculate the accuracy of the classifier
    return correct / len(test_data)
