#Модель для 3 параметров
import numpy as np
X = np.array([[-1, -1, 2], [-2, -1, 4], [1, 1, 5], [2, 1, 2]]) #Тут массив из 3
y = np.array([2, 1, 2, 2])
from sklearn.svm import SVC
clf = SVC()
clf.fit(X, y)
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
print(clf.predict([[-0.8, -1, 5]])) # И тут массив из 3