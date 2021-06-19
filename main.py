import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier


mammographic_masses_data = pd.read_csv('mammographic_masses_data.csv')
print(mammographic_masses_data)
cols = ['BI_RADS', 'age', 'shape', 'margin', 'density', 'severity']
mammographic_masses_data = pd.read_csv('mammographic_masses_data.csv', na_values=['?'], names=cols)
print(mammographic_masses_data.head())

print(mammographic_masses_data.describe())
print(mammographic_masses_data.loc[(mammographic_masses_data['age'].isnull()) |
                                   (mammographic_masses_data['shape'].isnull()) |
                                   (mammographic_masses_data['margin'].isnull()) |
                                   (mammographic_masses_data['density'].isnull())])
# Here, we got that 'NA' is distributed evenly
# in all coloumns, so will drop all rows with 'NA'

mammographic_masses_data.dropna(inplace=True)
print(mammographic_masses_data.describe())

X = mammographic_masses_data[['age', 'shape', 'margin', 'density']].values
y = mammographic_masses_data['severity'].values
print(X)
print(y)

scale = StandardScaler()
X = scale.fit_transform(X)
print(X)
print()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=5)

model1 = DecisionTreeClassifier()
cv_score_DT = cross_val_score(model1, X, y, cv=10)
print('Accuracy score of Decision Tree Classifier is : ', cv_score_DT.mean())

model2 = RandomForestClassifier(n_estimators=10)
cv_score_RF = cross_val_score(model2, X, y, cv=10)
print('Accuracy score of Random Forest Classifier is : ', cv_score_RF.mean())

model3 = KNeighborsClassifier(n_neighbors=7)  # from range 1 to 10, got best performance on n=7
cv_score_KNN = cross_val_score(model3, X, y, cv=10)
print('Accuracy score of K- Nearest Neighbours Classifier is : ', cv_score_KNN.mean())

model4 = MultinomialNB()
scalar = MinMaxScaler()
NB_X = scalar.fit_transform(X)
cv_score_NB = cross_val_score(model4, NB_X, y, cv=10)
print('Accuracy score of Naive Bayes Classifier is : ', cv_score_NB.mean())

model5 = SVC(kernel='rbf', C=1)  # out of all kernels, got best performance for 'rbf'
cv_score_svm = cross_val_score(model5, X, y, cv=10)
print('Accuracy score of Support Vector Machines Classifier is : ', cv_score_svm.mean())

model6 = LogisticRegression()
cv_score_LR = cross_val_score(model6, X, y, cv=10)
print('Accuracy score of Logistic Regression Classifier is : ', cv_score_LR.mean())

def createModel():
    model7= Sequential()
    model7.add(Dense(1, kernel_initializer='normal', activation='relu'))
    model7.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    model7.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model7
estimator= KerasClassifier(build_fn=createModel, epochs= 100, verbose= 0)
cv_score_NN= cross_val_score(estimator, X, y, cv=10)
print('Accuracy score of Neural Network Classifier is : ', cv_score_NN.mean())