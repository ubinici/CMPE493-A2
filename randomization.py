import random
from multinomial_nb import evaluate_multinomial_nb
from bernoulli_nb import evaluate_bernoulli_nb

# Function to run a single randomization trial
def run_randomization_trial(args):
    train_data, test_data, alpha = args
    test_data_copy = test_data.copy()
    random.shuffle(test_data_copy)
    m_macro_f_score = evaluate_multinomial_nb(train_data, test_data_copy, alpha)
    b_macro_f_score = evaluate_bernoulli_nb(train_data, test_data_copy, alpha)
    return m_macro_f_score - b_macro_f_score

# Function to perform a randomization test
def randomization_test(train_data, test_data, best_alpha_multinomial, best_alpha_bernoulli, num_trials=1000):
    # Calculate the observed difference in Macro F-scores
    _, observed_m_macro_f_score = evaluate_multinomial_nb(train_data, test_data, best_alpha_multinomial)
    _, observed_b_macro_f_score = evaluate_bernoulli_nb(train_data, test_data, best_alpha_bernoulli)
    observed_diff = abs(observed_m_macro_f_score - observed_b_macro_f_score)

    # Perform randomization test
    count = 0
    for _ in range(num_trials):
        permuted_test_data = list(test_data)
        random.shuffle(permuted_test_data)

        _, permuted_m_macro_f_score = evaluate_multinomial_nb(train_data, permuted_test_data, best_alpha_multinomial)
        _, permuted_b_macro_f_score = evaluate_bernoulli_nb(train_data, permuted_test_data, best_alpha_bernoulli)
        permuted_diff = abs(permuted_m_macro_f_score - permuted_b_macro_f_score)

        if permuted_diff >= observed_diff:
            count += 1

    return count / num_trials
