#Titanic Survival Predictions  

##Assumptions on Data
The assumptions of this data is that it is accurate and represents the entire
population of the titanic.However, there are less than 900 individuals listed in
this data and there were almost 3000 onboard the titanic. Therefor, we are
taking a risk in assuming that this population is representative and we are also
taking a risk in assuming that these old records are accurate.

##Penalty Type
L1 (lasso) penalties will reduce certain coefficients in a way that pushes
features influence towards zero, effectively eliminating the weight of some
features.
L2 (Ridge) penalties will generally rein in all coefficients so that they
do not overfit the model.
Using ridge is helpful if we are afraid of making our model too bias due to our
training data, while using lasso is helpful if we are unsure of which features
we should try to eliminate.

##Penalty Intensity
C is related to the amount of penalty we apply. C is the inverse of alpha. Alpha
is the multiplier we use to apply penalty. As alpha increases, more penalty is
added, and the model becomes more sensitive to larger coefficients

##Classification Threshold
If we change our threshold to .90, we would be much more confident that our
predicted survivals were true survivals. However, we would become less confident
that our deaths were true death. We would reduce our false positive rate, but
increase our false negative rate.

##Process
The data was obtained from a SQL database and then examined and cleaned in Python.
The purpose of this project is to predict if an individual survived the sinking
of the Titanic, so survival data was separated out and set as a target. The
correlation of each of the variables was then checked against survival. It turns
out that Sex and Fare have the highest correlation with survival Upon examination,
roughly 60% of the individuals in this dataset did not survive the disaster.
Therefore, the model we are attempting to build should have an accuracy higher
than 60% to be considered successful.

###Logist Regression
A simple logistic regression on the cleaned data produced a score of 79.9% accuracy.
However, there was no train test split or cross validation performed on this basic
model, so these results are not reliable are likely extremely bias.

####Validation
Next, a simple 80:20 train test split with cross validation of 12 folds was
performed and the accuracy of the model was tested. This model had an average
accuracy score of 76.2%. A confusion matrix was constructed for this test, which
shows that recall and precision are roughly equal for this model. An ROC curve
was also constructed with an AUC value of roughly 0.845.

####Grid Search
Next, a grid search with cross validation was performed in order to find the
optimal parameters for a this model. L1 and L2 penalties were considered as well
as a range of C values (logspace(-5,1,50)). L1 penalties with a C of 1.842
provided the best model with an accuracy of 75.5%.

###KNN
A simple K Nearest Neighbors model was constructed with the data. However, the
accuracy was only roughly 65.0%, which is only just better than the base accuracy.
Performing a grid search on this model with a range of n neighbors (from 1-54)
slightly increased this accuracy to 67.8%. The best KNN model used 31 neighbors.

####KNN Confusion Matrix vs LogReg Performance
Interestingly, when a confusion matrix was produced for the KNN model, the KNN
model noticeably better at distinguishing between true negatives and false
positives, but much worse at distinguishing between true positives and false
negatives. When the ROC curves for KNN and LogReg are plotted together, it is
clear that the Log Reg model is a better performing model. The AUC score for
LogReg is .845 and .677 for KNN.

####Complexity of KNN and Choosing between LogReg
As we use more neighbors, our model becomes more bias because it becomes more
complex. Logistic regression is usually a better choice than KNN because it is a
more sophisticated model and it requires less storage to run. KNN is a good model
if you are looking for something simple, the data set is not too large, and you
want as transparent of a model as possible.

###Grid Searching LogReg With Average Precision
When the grid search for LogReg was run with precision instead of accuracy, the
score of our model increased to 82.1%. However the confusion matrix confirmed
that the prediction breakdown was similar to the logistic regression but with
one more false positive and one less true negative. Even though this model was
built to minimize false negatives, the number of false negatives did not actually
change.
