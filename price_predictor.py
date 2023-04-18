import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


cars = pd.read_csv("dataset\Dataset.csv")
# cleaning the unnecessary columns / Data CLeaning
cars = cars.drop(['full_model_name','brand_rank', 
       'distance below 30k km', 'new and less used', 'inv_car_price',
       'inv_car_dist', 'inv_car_age', 'inv_brand', 'std_invprice',
       'std_invdistance_travelled', 'std_invrank', 'best_buy1', 'best_buy2'],axis=1)
# print(cars.columns)

# trimming outliers using interquartile range
percentile25 = cars['price'].quantile(0.25)
percentile75 = cars['price'].quantile(0.75)
iqr = percentile75 - percentile25
upper_limit = percentile75 + 1.5 * iqr
lower_limit = percentile25 - 1.5 * iqr
print('Highest allowed',upper_limit)
print('Lowest allowed',lower_limit)
print(cars[(cars['price'] > upper_limit) | (cars['price'] < lower_limit)])
car_index = cars[(cars['price'] > upper_limit) | (cars['price'] < lower_limit)].index
cars.drop(car_index,inplace=True) 
cars = cars.reset_index()
cars.drop('index',axis=1,inplace=True)
print(cars)


# assigning x and y values
y = cars.iloc[:,3]
# print(y)
x = cars.drop(['price'],axis=1)
# print(x.dtypes)

# encoding the categorical data
print(x.isnull())
for a in ['brand','model_name','fuel_type', 'city']:
       labelEncoder = LabelEncoder()
       x[a+'_enc'] = labelEncoder.fit_transform(x[a])
       print(labelEncoder.classes_)

       names = labelEncoder.classes_
       nam = []
       for b in names:
              if b.isnumeric():
                     b = b + '_model'
              nam.append(b)

       enc = OneHotEncoder(handle_unknown='ignore')
       enc_df = pd.DataFrame(enc.fit_transform(x[[a]]).toarray())
       enc_df = enc_df.set_axis(nam,axis=1)
       # print(enc_df.columns)
       x = x.join(enc_df,rsuffix='_other')
       # print(x)

# print(x.columns)
encoded = x[['brand','brand_enc', 'model_name','model_name_enc', 'fuel_type','fuel_type_enc', 'city','city_enc']]
x = x.drop(['brand','brand_enc', 'model_name','model_name_enc', 'fuel_type','fuel_type_enc', 'city','city_enc'],axis=1)
print(x.describe())
print(x)

# train test split
xTrain,xTest,yTrain,yTest = train_test_split(x,y,test_size=0.1)

# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 1000)

# Train the model on training data
rf.fit(xTrain, yTrain)

# Use the forest's predict method on the test data
predictions = rf.predict(xTest)
# Calculate the absolute errors
errors = abs(predictions - yTest)
# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2))

# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / yTest)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')


# Saving feature names for later use
x_list = list(x.columns)
# Get numerical feature importances
importances = list(rf.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 6)) for feature, importance in zip(x_list, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]


# Set the style
plt.style.use('fivethirtyeight')
# list of x locations for plotting
x_values = list(range(len(importances)))
# Make a bar chart
plt.bar(x_values, importances, orientation = 'vertical')
# Tick labels for x axis
plt.xticks(x_values, x_list, rotation='vertical')
# Axis labels and title
plt.ylabel('Importance')
plt.xlabel('Variable')
plt.title('Variable Importances')
plt.show()