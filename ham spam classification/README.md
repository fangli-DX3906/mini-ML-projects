## Introduction

I implement and test the Naive Bayes, Logistic, and Perceptron classifier for text classification.

For training Logistic classifier, I add L2 regularization to avoid overfitting.



## Code Structure

 Our code contains three classes: `ParseTextData`, `NBTextClassifier`, and `DiscriminativeTextClassifier`. 

### `ParseTextData`

This class the father class and is used for data parsing.

### `NBTextClassifier`

This class is used to train a Naive Bayes classifier. It is inherited from `ParseTextData`. 

When initializing an instance, no need to pass any parameters. However, when `fit()` is called, it will add two attributes: `clsPrior` and `condProb`. These two attributes are used to calculate the classification accuracy for later use. 

### `DiscriminativeTextClassifier`

This class is used to implement perceptron and Logistic classifier. It is also  inherited from `ParseTextData`. 

When initializing the class, we need to pass a param to tell if we want to train a perceptron or a logistic classifier. Two param values are allowed: Perceptron or Logistic. For the Perceptron classifier, labels are coded as $+1$ and $-1$ ,but labels are coded as $+1$ and $0$ for the Logistic regression.  
