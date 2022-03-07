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
Here the LightGBM method was applied to the same data set used for part 1. 

### Sources:
- https://www.kaggle.com/lasmith/house-price-regression-with-lightgbm
