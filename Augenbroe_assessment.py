# Programming assessment
# Anneke Augenbroe
import pandas as pd 
import numpy as np
import seaborn as sns
from scipy import stats
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import metrics
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor

## Read in csv file and look at input data
data = pd.read_csv('/Users/annekeaugenbroe/Desktop/Programming_Assessment/garments_worker_productivity.csv')
[n,m]=data.shape #n = number of training examples, m = number of attributes

print("Preprocessed data description:")
print(data.describe())

## Data Cleaning- Null Values
#checking for data type and missing values
data.dtypes
#date is object, changing to datetime64
data.date = pd.to_datetime(data.date)
data.isnull().sum()
#506 null values for work in progress; over 40% of n affected
sns.displot(data.wip)
#data is centered around the mean, but missing for finishing department,
#replace with 0
data.wip = data.wip.fillna(0)


## Data Cleaning- Outliers
#First, see how many outliers there are and see how much data would be lost 
#if I removed all outliers
newdata = data
for i in range(5,14): #for loop over numerical features
    outliers = n - data[(np.abs(stats.zscore(data.iloc[:,i])) < 3.0)].shape[0]
    #counts number of outliers for each feature
    newdata = newdata[(np.abs(stats.zscore(newdata.iloc[:,i])) < 3.0)]
    #removes outliers
loss = (n - newdata.shape[0])/n #9% loss, way too much
#After seeing that we would lose too much data and risk overfitting our 
#regression models, we look at a case by case basis
data = data[(np.abs(stats.zscore(data.iloc[:,5])) < 4)]
data = data[(np.abs(stats.zscore(data.iloc[:,8])) < 4)]
#targeted productivity of 0.07 in column 5, probably a typo, 
#also column 8 overtime 1 order of magnitude higher
#not touching rare events that may greatly affect productivity


#Feature Scaling
#numerical feature scaling using min-max scaling for numerical categories except 
#targeted_productivity because already on 0-1 scale
numerical = data.iloc[:,6:14]
for column in numerical.columns:
        numerical[column] = (numerical[column] - numerical[column].min()) / (numerical[column].max() - numerical[column].min())
numerical = pd.concat([data.iloc[:,5],numerical],axis=1,join='inner')

#Feature Encoding
categorical = data.iloc[:,1:5] #quarter, department, day, team
ohe = pd.get_dummies(data=categorical, columns=['quarter', 'department', 'day', 'team']) #one hot encoding
df = pd.concat([data.iloc[:,0],ohe,numerical,data.iloc[:,14]], axis=1, join="inner") #add date and actual productivility back in
#there were two columns for finishing department due to a space, need to combine
df.department_finishing=df.iloc[:,6]+df.iloc[:,7]
df = df.drop('department_finishing ', axis=1)
df = df.rename(columns={"department_sweing": "department_sewing"})
#correct sewing typo

#Making date suitable for modeling
df.index = df.date
df.index = df.index.to_julian_date() - 2457022.5
#date ranging from 1 to 70 days
df.date = df.index
df.date = (df['date'] - df['date'].min()) / (df['date'].max() - df['date'].min())
df = df.reset_index(drop=True)


#Describe processed data
print("Processed data description:")
print(df.describe())

#Correlation Heatmap
fig, ax = plt.subplots(figsize = (10,8))
sns.heatmap(numerical.corr(), annot = True, ax = ax)

#Data Partitioning
# split
x = df.iloc[:,0:35] #features
y =df['actual_productivity'] #labels
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)

#Linear Regression Model
lin =linear_model.LinearRegression() #from sklearn
lin.fit(x_train, y_train) #fit on training set
pred_lin = lin.predict(x_test) #prediction on test set
#visualization of linear regression model
plt.scatter(y_test, pred_lin, alpha=0.7, c='b') 
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
plt.xlabel("Productivity (Actual)")
plt.ylabel("Productivity (Predicted)")
plt.title("Linear Regression Model")
plt.show
#Evaluation of Linear Regression Model
evaluation = pd.DataFrame(columns = [ 'Model','Mean Squared Error','Mean Absolute Error', 'R2 Score'])
mse_lin=metrics.mean_squared_error(y_test,pred_lin)
mae_lin=metrics.mean_absolute_error(y_test, pred_lin)
r2_lin = metrics.r2_score(y_test,pred_lin)
lin_result = {'Model':lin,'Mean Squared Error':mse_lin,'Mean Absolute Error':mae_lin, 'R2 Score':r2_lin}
evaluation = evaluation.append(lin_result, ignore_index=True)

# Random Forest Regression
random_forest = RandomForestRegressor(n_estimators = 100 ,  random_state = 10)
random_forest.fit(x_train,y_train)
random_forest_pred = random_forest.predict(x_test)
plt.scatter(y_test, random_forest_pred, alpha=0.6, c='g') #visualization
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
plt.xlabel("Productivity (Actual)")
plt.ylabel("Productivity (Predicted)")
plt.title("Random Forest Model")
plt.show
evaluation_rf = pd.DataFrame(columns = [ 'Model','Mean Squared Error','Mean Absolute Error', 'R2 Score'])
mse_rf=metrics.mean_squared_error(y_test,random_forest_pred)
mae_rf=metrics.mean_absolute_error(y_test,random_forest_pred)
r2_rf = metrics.r2_score(y_test,random_forest_pred)
rf_result = {'Model':'Random Forest','Mean Squared Error':mse_rf,'Mean Absolute Error':mae_rf, 'R2 Score':r2_rf}
evaluation = evaluation.append(rf_result, ignore_index=True)

# Neural Network Regression 
mlp = MLPRegressor(solver='adam', hidden_layer_sizes=(37,),
                           max_iter=1000, activation='relu')
mlp.fit(x_train, y_train)
pred_nn = mlp.predict(x_test)

plt.scatter(y_test, pred_nn, alpha=0.6, c='r') #visualization
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
plt.xlabel("Productivity (Actual)")
plt.ylabel("Productivity (Predicted)")
plt.title("Neural Network Regression Model")
plt.show

mse_nn=metrics.mean_squared_error(y_test,pred_nn)
mae_nn=metrics.mean_absolute_error(y_test, pred_nn)
r2_nn = metrics.r2_score(y_test,pred_nn)
nn_result = {'Model':"Neural Network",'Mean Squared Error':mse_nn,'Mean Absolute Error':mae_nn, 'R2 Score':r2_nn}
evaluation = evaluation.append(nn_result, ignore_index=True)
nn_result

print(evaluation)
