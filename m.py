'''import pandas as pd # To manage data as data frames
import numpy as np # To manipulate data as arrays
from sklearn.linear_model import LogisticRegression

data = pd.read_csv('./fd_change-85.csv' , usecols=['Ad Supported','Fraud',	'In App Purchases',	'Rating','review'])

variety_mappings = {0: 'Safe', 1: 'Fraud'}

# Encoding the target variables to integers
#data = data.replace(['Safe','Fraud' ], [0, 1])

group1=['Ad Supported']
group2 =['In App Purchases','Rating','review']
group3 = ['Fraud']
new_col = group1+group2+group3
df2 = data[new_col]
df2.dropna(inplace=True)



X = df2.iloc[:, 0:-1] # Extracting the features/independent variables

y = df2.iloc[:, -1] # Extracting the target/dependent variable


from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(
        X, y, test_size = 0.25, random_state = 0)

from sklearn.preprocessing import StandardScaler

sc_x = StandardScaler()
xtrain = sc_x.fit_transform(xtrain)
xtest = sc_x.transform(xtest)


from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(xtrain, ytrain)
y_pred = classifier.predict(xtest)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(ytest, y_pred)

print("Confusion Matrix : \n", cm)


from sklearn.metrics import accuracy_score
print ("Accuracy : ", accuracy_score(ytest, y_pred))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import validation_curve
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
#
# IRIS Dataset is loaded



#
# Create training and test split
#

#
# Create the pipeline having steps for standardization and estimator as LogisticRegression
#
pipeline = make_pipeline(StandardScaler(), LogisticRegression(solver='lbfgs', penalty='l2', max_iter=10000, random_state=1))
#
# Get Training and test scores using validation curve method
# Pay attention to the parameter values range set as param_range
#
param_range = [0.001, 0.05, 0.1, 0.5, 1.0, 10.0]
train_scores, test_scores = validation_curve(estimator=pipeline,
                                             X=xtrain, y=ytrain,
                                             cv=10,
param_name='logisticregression__C', param_range=param_range)
#
# Find the mean of training and test scores out of 10-fod StratifiedKFold cross validation run as part fo execution of validation curve
#
train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
#
# Plot the model scores (accuracy) against the paramater range
#
plt.plot(param_range, train_mean,
         marker='o', markersize=5,
         color='blue', label='Training Accuracy')
plt.plot(param_range, test_mean,marker='o',
          markersize=5,
         color='green', label='Validation Accuracy')
plt.xlabel('Parameter C')
plt.ylabel('Accuracy')
plt.grid()
plt.show()'''


import pandas as pd # To manage data as data frames
import numpy as np # To manipulate data as arrays
from sklearn.linear_model import LogisticRegression



# importing necessary libraries
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

# loading the iris dataset
iris = pd.read_csv('./fd_change-85.csv' , usecols=['Ad Supported','Fraud',	'In App Purchases',	'Rating','review'])
group1=['Ad Supported']
group2 =['In App Purchases','Rating','review']
group3 = ['Fraud']
new_col = group1+group2+group3
df2 = iris[new_col]
df2.dropna(inplace=True)
variety_mappings = {0: 'Safe', 1: 'Fraud'}
# X -> features, y -> label
X = df2.iloc[:, 0:-1] # Extracting the features/independent variables

y = df2.iloc[:, -1] # Extracting the target/dependent variable

# dividing X, y into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)

# training a DescisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
dtree_model = DecisionTreeClassifier(max_depth = 2).fit(X_train, y_train)
dtree_predictions = dtree_model.predict(X_test)

# creating a confusion matrix
cm = confusion_matrix(y_test, dtree_predictions)

from sklearn.metrics import accuracy_score
print ("Accuracy : ", accuracy_score(y_test, dtree_predictions))



def classify(a, b, c, d):
    arr = np.array([a, b, c, d]) # Convert to numpy array
    arr = arr.astype(np.float64) # Change the data type to float
    query = arr.reshape(1, -1) # Reshape the array
    prediction = variety_mappings[dtree_model.predict(query)[0]] # Retrieve from dictionary
    return prediction








import numpy as np

from yellowbrick.model_selection import ValidationCurve

from sklearn.tree import DecisionTreeRegressor



viz = ValidationCurve(
    DecisionTreeRegressor(), param_name="max_depth",
    param_range=np.arange(1, 11), cv=10, scoring="r2"
)

# Fit and show the visualizer
viz.fit(X, y)
viz.show()
