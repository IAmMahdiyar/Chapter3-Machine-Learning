from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

face_data = fetch_lfw_people(min_faces_per_person=80)

X = face_data.data
Y = face_data.target
labels = face_data.target_names

print("Labels:", labels)

for i in range(5):
    print("Class", labels[i], ":", {(Y == i).sum()})

plt.show()

print("Fitting...")

x_train, x_test, y_train, y_test = train_test_split(X, Y)

svm = SVC(class_weight='balanced')

parameters = {'C': [0.1, 1, 10], 'gamma': [1e-07, 1e-08, 1e-06], 'kernel' : ['rbf', 'linear'] }

grid_search = GridSearchCV(svm, parameters, cv=5, n_jobs=-1)

grid_search.fit(x_train, y_train)

print("Best Settings for SVC:", grid_search.best_params_)

best_svm = grid_search.best_estimator_

print("Score:", best_svm.score(x_test, y_test))

pred = best_svm.predict(x_test)

print(classification_report(y_test, pred))