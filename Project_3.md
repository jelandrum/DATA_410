# Project 3: Concepts of Multivariate Regression Analysis and Gradient Boosting

This in an analysis of real data sets, one extracted from the 1974 *Motor Trend US* magazine and comprises fuel consumption and 10 aspects of automobile design and performance for 32 automobiles (1973-1974 models), and the other is a version of the Boston Housing Prices data that comes from the Scikit-Learn API. 

For the first analysis, the application of multivariate regression methods on the "Cars" data set were applied. Here, we considered engine, cylinder, and weight, as input variables ('ENG','CYL','WGT'). The output varable is the mileage, or 'MPG', for the "Cars" data set. In the second analysis, there are many more variables, but for this analysis the rooms, crime, and tax variables were the inputs. These variables represent the average number of rooms per house ('rooms'), crime per capita by town ('crime'), and value of property tax per ten thousand dollars ('tax'). The output of the Boston data was the corrected median value of owner-occupied housing in thousands, also known ad 'cmedv.'This is the procedure used for a multivariate multiple regression analysis, where the relationship between multiple input variables and one dependent variable is the focus. For each method the crossvalidated mean square error or residual error was collected.

## Extreme Gradient Boosting (xgboost)
Extreme Gradient Boosting, or XGB, is a analysis method that helps prevent overfitting. Regularization techniques in XBG uses both lambda and gamma hyperparameters. We then want to return only one output value, which makes us take the sum of the residual divided my the number of residuals plus lambda (sum(e)/e +lambda). This gives us the mean of the residuals for the method in the decision tree, the first prediction considering the rate our program is learning the pattern of the predefined relationship. Our cross-validated MSE is "the average of the differences between the predictions and the actual values squared" (Maklin).


## Cars Output:
```
The Cross-validated Mean Squared Error for LWR is : 16.927710396099975
The Cross-validated Mean Squared Error for Boosted LWR is : 16.74826832858372
The Cross-validated Mean Squared Error for XGB is : 16.559417572167884
```
The cars output shows that the lowest mean squared error belongs to the extreme gradient boosting method.

## Boston Housing Data Output
Initially, I replicated the XGBoost example outlined in the toward data science article by Cory Maklin: https://towardsdatascience.com/xgboost-python-example-42777d01001e. Upon replication, this method gave me the mse output of 9.495122199898898 which is different from the output in the example. I believe this is because of the improvements in our machine learning technology and packages over the last two years. Additionally, reproducing this model required that all of the features/variables were used in predicting the housing price which was our target variable in this case.
```
The Cross-validated Mean Squared Error for LWR is : 27.07174319855117
The Cross-validated Mean Squared Error for Boosted LWR is : 28.18167859624461
The Cross-validated Mean Squared Error for XGB is : 26.649022160777026
```
The Boston Housing Prices multivariate regression analysis proves, just as with Cars, that extreme gradient boosting is the best method fro this kind of analysis. It not only helps prevent overfitting but also reduces the mean squared error making it the better predictive modeling technique.
