import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.linear_model import LinearRegression, ElasticNet, BayesianRidge, OrthogonalMatchingPursuit, ARDRegression, PassiveAggressiveRegressor, RANSACRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor


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
# print('Highest allowed',upper_limit)
# print('Lowest allowed',lower_limit)
# print(cars[(cars['price'] > upper_limit) | (cars['price'] < lower_limit)])
car_index = cars[(cars['price'] > upper_limit) | (cars['price'] < lower_limit)].index
cars.drop(car_index,inplace=True) 
cars = cars.reset_index()
# print(cars)
cars.drop('index',axis=1,inplace=True)
# print(cars)


# assigning x and y values
y = cars.iloc[:,3]
# print(y)
x = cars.drop(['price'],axis=1)
# print(x.dtypes)
# print(x)
# encoding the categorical data
# print(x.isnull())
for a in ['brand','model_name','fuel_type', 'city']:
       labelEncoder = LabelEncoder()
       x[a+'_enc'] = labelEncoder.fit_transform(x[a])
       # print(labelEncoder.classes_)

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
# print(x.describe())
# print(x)

# train test split
xTrain,xTest,yTrain,yTest = train_test_split(x,y,test_size=0.2)
models = [LinearRegression(), RandomForestRegressor(n_estimators=1000), GradientBoostingRegressor(), ElasticNet(), BayesianRidge(), SVR(), OrthogonalMatchingPursuit(), ARDRegression(), RANSACRegressor(), DecisionTreeRegressor(), MLPRegressor(max_iter=1000), GaussianProcessRegressor(), ExtraTreesRegressor(n_estimators=1000)]
model_accuracy=[]
# Instantiate model with 1000 decision trees
# ml = RandomForestRegressor(n_estimators = 1000)
# ml = SVR()

for ml in models:
    
    # Train the model on training data
    ml.fit(xTrain, yTrain)

    # Use the forest's predict method on the test data
    predictions = ml.predict(xTest)

    # Calculate the absolute errors
    errors = abs(predictions - yTest)
    # Print out the mean absolute error (mae)
    print('Mean Absolute Error:', round(np.mean(errors), 2))

    # Calculate mean absolute percentage error (MAPE)
    mape = 100 * (errors / yTest)
    # Calculate and display accuracy
    accuracy = 100 - np.mean(mape)
    print('Accuracy:', round(accuracy, 2), '%.')
    model_accuracy.append(accuracy)
    print()
models = [ type(m).__name__ for m in models ]

accuracy_df = pd.DataFrame({"Model_name":models,"Accuracy":model_accuracy})
fig, ax = plt.subplots()
bar1 = ax.bar(models,model_accuracy, width=.6, zorder=1)
ax.grid(which="major", axis='x', color='#DAD8D7', alpha=0.5, zorder=0)
ax.grid(which="major", axis='y', color='#DAD8D7', alpha=0.7, zorder=0)
ax.set_axisbelow(True)
for i, rect in enumerate(bar1):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width() / 2, 1.5, models[i], ha='center', rotation=90)
plt.title("Model Accuracy")
# reformat x axis
ax.set_xticklabels([])
ax.set_xlabel('ML Models', fontsize=12)
ax.xaxis.set_label_position("bottom")
ax.xaxis.set_tick_params(pad=3, labelsize=12, labelrotation=90)

# Reformat y-axis
ax.set_yticks(np.arange(0,100,10))
ax.set_ylabel('Accuracy %', fontsize=12, labelpad=10)
ax.yaxis.set_label_position("left")
ax.yaxis.set_tick_params(pad=2, labeltop=False, labelbottom=True, bottom=False, labelsize=12)

# Add label on top of each bar
ax.bar_label(bar1, labels=[f'{e:,.1f}' for e in model_accuracy], padding=3, color='black', fontsize=8) 

# Colours - Choose the extreme colours of the colour map
colours = ["#bbdefb","#1a33d6"]

# Colormap - Build the colour maps
cmap = mpl.colors.LinearSegmentedColormap.from_list("colour_map", colours, N=256)
norm = mpl.colors.Normalize(0,100) # linearly normalizes data into the [0.0, 1.0] interval

# Plot bars
bar1 = ax.bar(models, model_accuracy, color=cmap(norm(model_accuracy)), width=0.6, zorder=2)

average = np.asarray(model_accuracy).mean()

plt.axhline(y=average, color = 'grey', linewidth=3)

# Determine the y-limits of the plot
ymin, ymax = ax.get_ylim()
# Calculate a suitable y position for the text label
y_pos = average/ymax + 0.03
# Annotate the average line
ax.text(0.88, y_pos, f'Average = {average:.1f}', ha='right', va='center', transform=ax.transAxes, size=8, zorder=3)
print(accuracy_df)
plt.show()

'''
# Saving feature names for later use
x_list = list(x.columns)
# Get numerical feature importances
importances = list(ml.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 6)) for feature, importance in zip(x_list, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]
'''

