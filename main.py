import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, r2_score
print(plt.style.use('dark_background'))

df = pd.read_csv('diabetes.csv')

# features and predicted columns
df_features = df.drop('Outcome', axis=1)
X = df_features
y = df['Outcome']

# train test split
TS = float(input('enter test size:'))
if TS > 1:
    TS /= 100

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TS)

# global variables
error_rate = []
multi_trials = []
values = []
k = 0
predicted = 0
pred_log = 0

# determines best 'K' value
def elbow_method():
    global error_rate, multi_trials, values, k
    for value in range(1, 50):

        for i in range(1, 11):
            knn = KNeighborsClassifier(n_neighbors=value)
            knn.fit(X_train, y_train)
            pred_value = knn.predict(X_test)
            multi_trials.append(np.mean(pred_value != y_test))

            if len(multi_trials) == 10:
                values.append(value)
                error_rate.append((sum(multi_trials) / len(multi_trials), value))
                multi_trials = []
            else:
                continue
    
    # manipulates K
    int(k)
    k = min(error_rate)[1]
    if k % 2 == 0:
        k -= 1
    print(f'Best K Value: {k} with error rate of {min(error_rate)[0].round(4) * 100}%')

# does KNN with best K value
def apply_best():
    global predicted
    knn = KNeighborsClassifier(n_neighbors=11)
    knn.fit(X_train, y_train)
    predicted = knn.predict(X_test)

# displays metrics
def knn_metrics():
    print(f'''
------------- KNN Metrics -------------
{'Confusion Matrix'}: \n {confusion_matrix(y_test, predicted)}
-------------
{'Classification Report'}: \n{classification_report(y_test, predicted)}
''')

# uses logistic regression
def log_regression():
    global pred_log
    log = LogisticRegression()
    log.fit(X_train, y_train)
    pred_log = log.predict(X_test)

# logistic regression metrics
def log_metrics():
    print(f'''
------------- Logistic Regression Metrics -------------
{'R^2 Score'}: \n {r2_score(y_test, pred_log)}
-------------
{'Confusion Matrix'}: \n {confusion_matrix(y_test, pred_log)}
-------------
{'Classification Report'}: \n{classification_report(y_test, pred_log)}
''')


elbow_method()
apply_best()
knn_metrics()
log_regression()
log_metrics()
