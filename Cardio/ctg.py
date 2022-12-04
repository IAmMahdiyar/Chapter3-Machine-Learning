import pandas
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report

#Current Directory Should be the same as ctg.py
df = pandas.read_excel("CTG.xls", "Raw Data")

X = df.iloc[1:2126, 3:-2].values
Y = df.iloc[1:2126, -1].values

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

svc = SVC(kernel='rbf')
parameters = {'C': (100, 1e3, 1e4, 1e5), 'gamma': (1e-08, 1e-7, 1e-6, 1e-5)}

grid_search = GridSearchCV(svc, parameters, n_jobs=-1, cv=5)

grid_search.fit(x_train, y_train)

best = grid_search.best_estimator_

print("Score:", best.score(x_test, y_test))

pred = best.predict(x_test)

print(classification_report(y_test, pred))