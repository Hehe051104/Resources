from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score

data = load_iris()
X = data.data
y = data.target


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)

y_pred_svm = svm_model.predict(X_test)


svm_accuracy = accuracy_score(y_test, y_pred_svm)
print(f"SVM模型的准确率: {svm_accuracy:.2f}")


regression_model = LinearRegression()
regression_model.fit(X_train, y_train)

y_pred_reg = regression_model.predict(X_test)

print("回归模型的预测值:", y_pred_reg)