# Credit Risk Resampling Report

## Overview of the Analysis

The purpose of this analysis was to use various techniques to train and evaluate models with imbalanced classes. In this analysis, we focused on using a Logistics Regression Model to compare both our original test data with our resampled test data.

The dataset (lending_data.csv) shows us historical lending activity from a peer-to-peer lending service companies that will allow us to build a model that can identify the creditworthiness of borrowers. Using the loan_status of 0 or 1 to show a 'healthy' loan vs a 'high-risk' loan as our main predicted value, we use the other values as predictors.

The main variable of prediction was how many healthy loans and high-risk loans which were labeled as '0' and '1' respectively. In order to determine how much of each we had within that y-variable (loan_status), we used value_counts, which gave us a count of each label within the data.

In this machine learning process, we had to teach the machine what was considered 'healthy' and what was considered 'high-risk' from the training data to allow it to take random data and predict out with a certain accuracy going forward what was 'healthly' and what was 'high-risk'. However, we noticed from the dataset that there were more 'healthy' loans than 'high-risk' loans and that they were easily outnumbered, which made obtaining a high accuracy for the 'high-risk' loans more difficult that the 'healthy' loans.

But apart from that, the two models we used during this machine learning process was the original Logistic Regression model with the train/test data, as well as the resampled data using oversampling in an effort to increase the number of 'high-risk' loan labels.


## Results

* Machine Learning Model 1:
  * Balance Accuracy = 95.2% -- the model was overall 95.2% accurate in predicting the labels as 'healthy' or 'high-risk' based on the X values
  * Precision -- 100% for 'healthy' loans and 85% for 'high-risk' loans -- the model measures 'healthy' loans at 100% precision, whereas it only measures 'high-risk' loans at 85% precision, would could be due to high-accuracy loans being so outnumbered by the healthy loans in the dataset, giving the machine less to learn from given the dataset
  * Recall -- 99% for 'healthy' loans and 91% for 'high-risk' loans -- which shows that our model is 99% good at correctly predicting healthy loans, and 91% for high-risk

* Machine Learning Model 2:
  * Balance Accuracy = 99.4% -- the model was overall 95.2% accurate in predicting the labels as 'healthy' or 'high-risk' based on the X values
  * Precision -- 100% for 'healthy' loans and 84% for 'high-risk' loans -- the model measures 'healthy' loans at 100% precision, whereas it only measures 'high-risk' loans at 84% precision, would could be due to high-accuracy loans being so outnumbered by the healthy loans in the dataset, giving the machine less to learn from given the dataset
  * Recall -- 99% for 'healthy' loans and 99% for 'high-risk' loans -- which shows that our model is 99% good at correctly predicting healthy loans, and 99% for high-risk


## Summary

In summary, both machine learning models have a high balance accuracy, showing that both models are pretty accurate in predicting the status of the loan based on the various factors (X). However, it seems that because of the outnumbering, once we do a resampling with an oversampling that the second model performs better even though status wise the precision for the high-risk loans decreases by 1 percentage point. This is because the balanced accuracy goes from 95.2% to 99.4%, showing an overall increase in the accuracy of the second model. The recall for the high-risk loan is also higher, meaning that the second model is better at correctly predicting high-risk loans over the first model. Which is important because performance does depend on the problem we are trying to solve. It is more important for us to be able to predict high-risk loans than healthy loans to the nature of it and risk associated with it. 
