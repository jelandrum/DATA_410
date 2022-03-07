## Multiple Boosting and Light GBM
### Part One: Implement multiple boosting algortihm and apply it to combinations of different regressors 
In this exercise, the "Concrete Compressive Strength" dataset was used. The imputs of this multiple boosting algorithm were 'concrete','water', and 'age'. These regressors were boosted a couple of times using the "Concrete Compressive Strength" dataset. 


The combination of multiple regression boosting methods that achieved the best cross-validated results is shown below. Other inputs were tested, however the boosting and obtaining cross-validated results took 20 minutes to run through for each trial, making this is a very time consuming process. 
```
yhat_blwr = boosted_lwr(xtrain,ytrain,xtest,Tricubic,1,True,model_boosting,2)

model_xgb = xgb.XGBRegressor(objective ='reg:squarederror',n_estimators=100,reg_lambda=20,alpha=1,gamma=10,max_depth=1)
model_xgb.fit(xtrain,ytrain)
yhat_xgb = model_xgb.predict(xtest)
```
The cross-validated results tell us that the boosted linear regression method produced a smaller error value (mse) when comparing the actual versus predicted y values, also known as concrete strenth in this exercise.
```
The Cross-validated Mean Squared Error for Boosted LWR is : 61.69791982682467
The Cross-validated Mean Squared Error for XGB is : 76.07840604775828
```

### Part Two: The LightGBM algorithm 
Here the LightGBM method was applied to the same data set used for part 1. I used the "House Price Regression with LightGBM" example by Lawrence Smith on Kaggle to implement the LightGBM method on the concrete dataset. Once again our predicitve values for x (i.e. age, water, and concrete) were used to predict y, our target variable of concrete strength. LightGBM is a custom function that enables powerful boosting implementation using gradient boosting with decision trees. There are numerous hyperparameters that are available for tweaking using LightBGM. In this implementation, the hyperparameters specified were: boosting_type = 'gbdt',n_estimators=100, max_depth= 1. For the rest of the hyperparameters, the default was accepted, such as the learning rate of 0.1. 

Ouput for the LightGBM algorithm:
```
The rmse of prediction is: 0.26938
```
In this exercise, the LightGBM model outperformed the other multiple regression models of XGBoost  and boosted locally weighted regression. For my hardware, the LGBM algorithm ran much faster than the other models, making it not only a high accuracy but also high speed model. This implementation can be an extremely helpful tool to refer to in the future for multiple regression methods. 

### Sources:
- https://www.kaggle.com/lasmith/house-price-regression-with-lightgbm
