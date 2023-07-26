# Fuzzy Genetic Algorithm for Text Classification
 This repository contains Python code for a Fuzzy Genetic Algorithm Spam Classifier. The classifier uses a combination of fuzzy logic and genetic algorithm to classify spam and legitimate messages. This Classifier is built using scikit-learn's TfidfVectorizer and PCA for dimensionality reduction.

 ## Installatioin
  1. Clone the repository to your local machine:
  ```bash
  git clone https://github.com/Ali-Pourgheysari/CI-phase-3-fuzzy-inference-system.git
  ```
  2. Install the required packages:
  ```bash
  pip install numpy pandas scikit-learn matplotlib
  ```

## Usage 
1. Make sure you have the dataset "SMSSpamCollection" in the same directory as the code.
2. Run the main script to preprocess the data, apply Fuzzy Genetic Algorithm classification, and evaluate the accuracy:
```bash
python main.py
```
3. The script will output the accuracy of the classifier and plot the fitness score over generations.

## Features
1. Fuzzy Functions: Contains various membership functions like sigmoid, gaussian, triangular, and trapezius used for fuzzy logic.
2. Rule Class: Represents a single rule in the Fuzzy Genetic Algorithm classifier, consisting of if_terms and a class_label.
3. Fuzzy_functions Class: Implements fuzzy logic operations and tests rules against input data.
4. data_preprocessing Class: Handles data preprocessing, including tokenization and dimensionality reduction using PCA.
5. genetic_algorithm Class: Implements the genetic algorithm to generate and evolve fuzzy rules for classification.
6. TfidfVectorizer: Converts text data into a numerical feature matrix using TF-IDF vectorization.
7. PCA: Performs dimensionality reduction using Principal Component Analysis.

## Contributing
Contributions to improve the classifier or add new features are welcome! Feel free to open a pull request with your changes.

Good Luck!