import collections
from data_processing import load_stopwords, load_data, preprocess_data, write_data_to_file, get_top_n_topics, split_data
from multinomial_nb import train_multinomial_nb, evaluate_multinomial_nb, test_multinomial_nb
from bernoulli_nb import train_bernoulli_nb, evaluate_bernoulli_nb, test_bernoulli_nb
from randomization import randomization_test

if __name__ == "__main__":
    # Load data and stopwords
    data_path = 'reuters21578/reuters21578'
    stopwords_path = 'stopwords.txt'
    raw_data, data = load_data(data_path)
    stopwords = load_stopwords(stopwords_path)
    
    # Preprocess data and retrieve top 10 topics along with label counts
    preprocessed_data = preprocess_data(data, stopwords)
    top_10_topics, multi_label_count = get_top_n_topics(preprocessed_data, 10)

    # Write preprocessed data to a file
    output_path = 'preprocessed_data.txt'
    write_data_to_file(preprocessed_data, output_path)

    # Filter articles belonging to one or more of the top 10 topics
    filtered_data = [(topics, tokens) for topics, tokens in preprocessed_data if set(topics).intersection(top_10_topics)]

    # Split the data into training, development, and test sets
    train_data, dev_data, test_data = split_data(raw_data, preprocessed_data)

    # Print the results
    print(f"\nTop 10 topics: {top_10_topics}")
    print(f"\nTraining set size: {len(train_data)}")
    print(f"Development set size: {len(dev_data)}")
    print(f"Test set size: {len(test_data)}")

    # Print the total number of vocabulary
    vocab = {token for _, tokens in preprocessed_data for token in tokens}
    print(f'Total number of vocabulary: {len(vocab)}')

    # Print the top 10 classes and their counts in the training and test sets
    train_data, dev_data, test_data = split_data(raw_data, preprocessed_data)
    train_counter = collections.Counter([topic for topics, _ in train_data for topic in topics])
    test_counter = collections.Counter([topic for topics, _ in test_data for topic in topics])
    print("\nTop 10 classes and their counts in the training and test sets:")
    for topic in top_10_topics:
        train_count = train_counter[topic]
        test_count = test_counter[topic]
        print(f"{topic}: Training set count: {train_count}, Test set count: {test_count}")

    # Print the total number of documents in the training and test sets
    print(f"\nTotal number of documents in the training set: {len(train_data)}")
    print(f"Total number of documents in the test set: {len(test_data)}")

    # Print the number of documents labeled with more than one of the top 10 classes
    print(f"\nNumber of documents labeled with more than one of the top 10 classes: {multi_label_count}\n")

    highest_macro_f_multinomial = -1
    best_alpha_multinomial = -1
    highest_macro_f_bernoulli = -1
    best_alpha_bernoulli = -1

    # Evaluate the Multinomial NB classifier with different alpha values
    for alpha in [1, 0.5, 2]:
        micro_f_score, macro_f_score = evaluate_multinomial_nb(train_data, dev_data, alpha)
        if macro_f_score > highest_macro_f_multinomial:
            highest_macro_f_multinomial = macro_f_score
            best_alpha_multinomial = alpha
        print(f"Alpha: {alpha}, Micro F-score (Multinomial): {micro_f_score}, Macro F-score (Multinomial): {macro_f_score}")

    # Evaluate the Bernoulli NB classifier with different alpha values
    for alpha in [1, 0.5, 2]:
        micro_f_score, macro_f_score = evaluate_bernoulli_nb(train_data, dev_data, alpha)
        if macro_f_score > highest_macro_f_bernoulli:
            highest_macro_f_bernoulli = macro_f_score
            best_alpha_bernoulli = alpha
        print(f"Alpha: {alpha}, Micro F-score (Bernoulli): {micro_f_score}, Macro F-score (Bernoulli): {macro_f_score}")

    multinomial_priors, multinomial_likelihoods = train_multinomial_nb(train_data, alpha=best_alpha_multinomial)
    bernoulli_priors, bernoulli_likelihoods = train_bernoulli_nb(train_data, alpha=best_alpha_bernoulli)

    multinomial_accuracy = test_multinomial_nb(test_data, multinomial_priors, multinomial_likelihoods)
    bernoulli_accuracy = test_bernoulli_nb(test_data, bernoulli_priors, bernoulli_likelihoods)

    print(f"Multinomial test set accuracy: {multinomial_accuracy}")
    print(f"Bernoulli test set accuracy: {bernoulli_accuracy}")

    num_trials = 10  # Set the number of trials for the randomization test
    p_value = randomization_test(train_data, test_data, best_alpha_multinomial, best_alpha_bernoulli, num_trials=num_trials)
    print(f"Randomization test p-value: {p_value}")


