import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB  
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier  
import joblib



long_data = pd.read_csv("data/oasis_longitudinal.csv")
long_data["SES"].fillna(long_data["SES"].mean(), inplace = True)
long_data["MMSE"].fillna(long_data["MMSE"].mean(), inplace = True)
long_data.drop(columns = ["Subject ID","MRI ID","Hand","Visit"], inplace = True)
long_data.rename(columns={"M/F" : "Gender"}, inplace=True)
long_data["Gender"] = np.where(long_data["Gender"]=="F", 1, 0)
long_data["Group"] = np.where(long_data["Group"] == 'Demented', 1, 0)

X = long_data.iloc[:, 1:]
Y = long_data.iloc[:,0:1]

sc = StandardScaler()
X_transformed = sc.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.20, random_state = 42)

#model objects
rf_model = RandomForestClassifier()
lr_model = LogisticRegression()
gb_model = GradientBoostingClassifier()
nb_model = GaussianNB()
svc_model = SVC()
dec_model= DecisionTreeClassifier(criterion='entropy', random_state=0)  

#model training
rf_model.fit(X_train, y_train)
lr_model.fit(X_train, y_train)
gb_model.fit(X_train, y_train)
nb_model.fit(X_train, y_train)
svc_model.fit(X_train, y_train)
dec_model.fit(X_train, y_train)

#predictions
rf_predictions = rf_model.predict(X_test)
lr_predictions = lr_model.predict(X_test)
gb_predictions = gb_model.predict(X_test)
nb_predictions = nb_model.predict(X_test)
svc_predictions = svc_model.predict(X_test)
dec_predictions = dec_model.predict(X_test)
#print(predictions)

print("Random Forest Classifier : {:.2%}".format(accuracy_score(y_test, rf_predictions)))
print("Logistic Regression : {:.2%}".format(accuracy_score(y_test, lr_predictions)))
print("Gradient Boosting Classifier : {:.2%}".format(accuracy_score(y_test, gb_predictions)))
print("Naive Bayes Classifier : {:.2%}".format(accuracy_score(y_test, nb_predictions)))
print("SVC Classifier : {:.2%}".format(accuracy_score(y_test, svc_predictions)))
print("Decision Tree Classifier : {:.2%}".format(accuracy_score(y_test, dec_predictions)))

#Naive Bayes model chosen
joblib.dump(nb_model, "nb_model.sav")