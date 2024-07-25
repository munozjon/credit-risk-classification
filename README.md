# credit-risk-classification

## Module 20 Challenge Report

### Overview of the Analysis

For this challenge, I analyzed loan risk data via a dataset of historical lending activity. This was provided by a peer-to-peer lending services company. The objective was to build a machine learning model that identifies the creditworthiness of borrowers. The labels for the model was provided by the *loan status* column of the dataset. This represented the status of the loan, with 0 being healthy and 1 meaning it has a high risk of defaulting. The features included *loan size*, *interest rate*, *borrower income*, *debt-to-income ratio*, *number of accounts*, *derogatory marks*, and *total debt*. These features describe the financial status of each borrower, and they will be used in the model to train the dataset to provide attributes for each *loan status* label in order for the model to predict.

For this challenge, a supervised model was used to fit, predict, and evaluate the data. I first read the CSV file to a pandas DataFrame. I then split the columns into the labels (y) and the features (X), as mentioned above. After, I split the data into training and testing datasets with the train_test_split function from SKLearn. Within this function, I pass a few parameters: I ensure the rows are shuffled, I specify the size of the test data to be 30% of the dataset, and the stratify parameter will ensure the same percentage of 0's and 1's from the dataset exist in the y_train and y_test splits. Additionally, I pass the random_state parameter for reproducability.

Following this, I fit the training data to a __logistic regression model__ using the LogisticRegression module from SKLearn. This will allow our data to properly predict our categorical target variable. After, I run the predict function on the model and use the X_test data as the parameter to test the model. Using these predictions, I create a confusion matrix along with the y_test data, and run the classification report for the model to determine its effectiveness.

### Results

* Machine Learning Model 1:

    * Accuracy: The accuracy score was 0.99, meaning the model was correct 99% of the time.

    * Precision: For the '0' label, the precision ratio was 1.00. This means that 100% of the predicted 0s were actually 0. Meanwhile, the precision ratio for the '1' label was 0.87, meaning just 87% of the predicted 1s were actually 1. In the context of the data, healthy loans were correctly predicted 100% of the time, while high-risk loans were precisely predicted 87% of the time.

    * Recall: For the '0' label, the recall ratio was 1.00, meaning none of the predicted 0s were incorrectly labeled as 1. Meanwhile, the recall ratio for the '1' label was 0.91, meaning about 9% of actual 1s were incorrectly predicted as 0s. In the context of the data, the predicted health loans were not wrongly categorized as high-risk. On the other hand, 9% of actual high-risk loans were incorrectly labeled as a healthy loan.

### Summary

In conclusion, the machine learning model trained and tested for this challenge proved to be a sufficient model for predicting loan risk status for borrowers. Based on the accuracy, precision, and recall scores, the model predicts both categories well. At the same time, the precision and recall scores for high-risk loans were slightly lower than what may be ideal for a loan risk predicting model. Because the 1s, or high-risk loans, are more important to predict, the precision and recall ratios should ideally be higher, so as to not incorrectly miss high-risk loans or wrongly categorize them as healthy ones. This could be due to the original dataset containing very few high-risk loans (2500) relative to the healthy loans (75036). Further training on the model with more high-risk loans would be beneficial in creating a better predicting model. As it is, however, it may be sufficient enough to recommend for use, as potential users may find the ratio to be okay for their particular use case.