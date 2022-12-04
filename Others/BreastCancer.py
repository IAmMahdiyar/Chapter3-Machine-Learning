from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

cancer_data = load_breast_cancer()

X = cancer_data.data
Y = cancer_data.target

n_pos = (Y == 1).sum()
n_neg = (Y == 0).sum()

print(n_pos ,"Positive Samples and", n_neg, "Negetive Samples out of", len(cancer_data.data))

x_train, x_test, y_train, y_test = train_test_split(X, Y)

svc = SVC(kernel="linear")

svc.fit(x_train, y_train)

score = svc.score(x_test, y_test)

print("score:", score)