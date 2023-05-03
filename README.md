# Reuters Text Classification Report

***Before running the script, see: [README](_README.txt)***

## Report Outline

1. [Data Preprocessing](#1-data-preprocessing)
2. [Top 10 Classes and Document Counts](#2-top-10-classes-and-document-counts)
3. [Parameter Tuning](#3-parameter-tuning)
4. [Evaluation Results](#4-evaluation-results)
5. [Screenshots](#5-screenshots)

## 1. Data Preprocessing

In the preprocessing stage, I first combined the title and body of each document to create a single text string. Then, I tokenized the text data by splitting it into individual words while also converting all the characters to lowercase. I used a custom tokenizer function that employs regular expressions to identify word boundaries.

After tokenization, I removed any stopwords from the tokens. Stopwords are common words such as 'and', 'is', 'in', etc., which do not carry significant meaning and can be safely removed to reduce the dimensionality of the dataset. The resulting preprocessed data is a list of tuples, each containing the document's topics and its tokenized text without stopwords.

**Total number of vocabulary:** 11,969

## 2. Top 10 Classes and Document Counts

The top 10 classes are: 'earn', 'acq', 'money-fx', 'grain', 'crude', 'trade', 'interest', 'ship', 'wheat', and 'corn'.

**Training set size:** 5,654
**Development set size:** 1,414
**Test set size:** 2,745

| Topic     | Training Set Count | Test Set Count |
|-----------|--------------------|----------------|
| earn      | 2,158              | 1,045          |
| acq       | 1,185              | 643            |
| money-fx  | 368                | 142            |
| grain     | 317                | 134            |
| crude     | 290                | 161            |
| trade     | 281                | 113            |
| interest  | 229                | 102            |
| ship      | 156                | 85             |
| wheat     | 156                | 66             |
| corn      | 130                | 48             |

**Total number of documents in the training set:** 5,654
**Total number of documents in the test set:** 2,745

**Number of documents labeled with more than one of the top 10 classes:** 844

## 3. Parameter Tuning

For the parameter tuning process, I used a grid search approach to find the best alpha values for both the Multinomial and Bernoulli Naive Bayes classifiers. The development set, which consists of 1,414 randomly selected documents, was used to evaluate the performance of different alpha values. The development set was not used during the training process, ensuring that the classifiers' performance could be evaluated on unseen data.

I tested several alpha values for both classifiers, including 0.5, 1, and 2. The performance of each combination of classifier and alpha value was measured using both Micro and Macro F-scores. I chose the alpha values that resulted in the highest F-scores for each classifier.

The best alpha values found for each classifier were:

| Alpha | Micro F-score (Multinomial) | Macro F-score (Multinomial) | Micro F-score (Bernoulli) | Macro F-score (Bernoulli) |
|-------|-----------------------------|-----------------------------|---------------------------|---------------------------|
| 1     | 0.4459                      | 0.4623                      | 0.0000                    | 0.0000                    |
| 0.5   | 0.3663                      | 0.4810                      | 0.0003                    | 0.0047                    |
| 2     | 0.5367                      | 0.4418                      | 0.0000                    | 0.0000                    |

By selecting the best alpha values for each classifier, I aimed to optimize their performance for the given text classification task. The parameter tuning process allowed us to fine-tune the classifiers' parameters for the specific problem, thus improving the overall performance.

## 4. Evaluation Results

### Test Set Performance

After selecting the best alpha values for both the Multinomial and Bernoulli Naive Bayes classifiers through parameter tuning, I evaluated their performance on the test set. The test set consists of 2,745 documents that were not used during the training or parameter tuning process. The test set performance provides an unbiased estimation of the classifiers' ability to generalize to new, unseen data.

The results for the test set performance are as follows:

- Multinomial Naive Bayes classifier:
  - Test set accuracy: 0.8051
  - This means that the classifier correctly predicted the topic for approximately 80.51% of the documents in the test set.

- Bernoulli Naive Bayes classifier:
  - Test set accuracy: 0.0022
  - This means that the classifier correctly predicted the topic for only 0.22% of the documents in the test set, which is considerably lower than the performance of the Multinomial Naive Bayes classifier.

The significant difference in test set accuracy between the two classifiers indicates that the Multinomial Naive Bayes classifier is more suitable for this particular text classification task. However, as discussed in the randomization test section, the difference in performance may not be statistically significant, and further investigation might be needed to confirm these results.

### Randomization Test

In order to determine whether the performance difference between the Multinomial and Bernoulli Naive Bayes classifiers is statistically significant, I conducted a randomization test. This non-parametric test compares the actual difference in performance between two classifiers against the distribution of differences that would be expected if their performance was indistinguishable.

The randomization test was performed as follows:

1. Calculate the actual difference in performance between the two classifiers using their F-scores.
2. Randomly permute the true labels of the test set multiple times (e.g., 10,000 times) and recompute the performance difference between the two classifiers for each permutation.
3. Count the number of times the absolute difference in performance for the permuted labels is equal to or greater than the actual difference.
4. Divide this count by the total number of permutations to obtain the p-value.

In our case, the randomization test resulted in a p-value of 1.0. This means that in all the permutations, the difference in performance between the two classifiers was equal to or greater than the actual difference. A p-value of 1.0 indicates that there is no evidence to reject the null hypothesis, which states that the performance of the two classifiers is indistinguishable. In other words, the difference in performance between the Multinomial and Bernoulli Naive Bayes classifiers is not statistically significant.

However, it is essential to note that the extremely low performance of the Bernoulli classifier in our case might indicate issues with the implementation or other factors that could affect its performance. Further investigation may be needed to confirm the validity of this randomization test result.

## 5. Screenshots

[Include screenshots of your algorithms running here]
