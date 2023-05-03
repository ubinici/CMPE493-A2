================================================================================
README - Text Classifier via Multinomial NB & Bernoulli NB Algorithms
================================================================================

This program consists of five Python scripts:

main.py           :    This script is responsible for loading data, 
                       preprocessing data, and training and evaluating a 
                       Multinomial NB and Bernoulli NB classifier.

data_processing.py:    This script loads data from a file and preprocess it.

multinomial_nb.py :    This script trains and evaluates a Multinomial NB 
                       classifier.

bernoulli_nb.py   :    This script trains and evaluates a Bernoulli NB 
                       classifier.

randomization.py  :    This script performs a randomization test on the 
                       results of the Multinomial NB and Bernoulli NB 
                       classifiers.

Python Version    :    3.8 or higher is recommended.

Instructions for Running the Program:
--------------------------------------------------------------------------------

1. Ensure that you have Python 3.8 or higher installed on your system. To check
   the installed version, open a terminal or command prompt and run:

   python --version

2. Install the required packages if necessary.

3. Place the main.py, data_processing.py, multinomial_nb.py, bernoulli_nb.py, 
   and randomization.py scripts in the same directory.

4. Run the main.py script to train and evaluate the Multinomial NB and Bernoulli 
   NB classifiers. In the terminal or command prompt, navigate to the directory 
   containing the scripts and run:

   python main.py

   This will print the results of the training and evaluation.

Note: Make sure that the document collection is in the "reuters21578/reuters21578"
      directory, or modify the script accordingly to point to the correct
      directory.

