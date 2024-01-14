
### Movie Sentiment Classification with N-Gram Language Model

This repository contains a Python implementation for sentiment classification using an N-Gram language model. Follow the steps below to reproduce the results:

#### Step 1: Import Movie Reviews

Import the movie reviews from the "Movie_Reviews.txt" dataset, ensuring that both positive and negative reviews are included in the dataset.

#### Step 2: Pre-process Text Data

Pre-process the text data by tokenizing, removing punctuation, and performing any other necessary steps to clean the input data.

#### Step 3: Choose N-Gram Model

Choose an appropriate value for N (e.g., bigram, trigram) and implement the N-Gram model using Python or a similar programming language. Explain your choice of N in the code comments or documentation.

#### Step 4: Calculate N-Gram Probabilities

Calculate N-Gram probabilities for each N-Gram in the corpus. This step involves computing the probability of each N-Gram given the training data.

#### Step 5: Calculate N-Gram Probability for Test Review

Calculate the N-Gram probability for the provided test movie review with respect to both positive and negative datasets. The test review is as follows:

```plaintext
“It's clear that the movie has both its enthusiasts and critics. While it may not be to everyone's taste, it's worth watching with an open mind to form your own opinion.”
```

#### Step 6: Predict Sentiment Category

Predict the category (positive or negative) of the test movie review based on the calculated N-Gram probabilities. Include an explanation of your prediction in the code comments or documentation.

#### Step 7: Understanding Perplexity

Explain the concept of perplexity and how it measures the model's performance in language modeling. Discuss how perplexity is calculated and interpreted in the context of N-Gram language models.
