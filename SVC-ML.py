#Модель для 3 параметров
import numpy as np
X = np.array([[-1, -1, 2], [-2, -1, 4], [1, 1, 5], [2, 1, 2]]) #Тут массив из 3
y = np.array([2, 2, 3, 2])
from sklearn.svm import SVC
clf = SVC(probability=True) #Обязательно задать для расчета вероятностей
clf.fit(X, y)

SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=True, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
print(clf.classes_)
print(clf.predict([[-1, -1, 2]])) # И тут массив из 3
print(clf.predict_proba([[-1, -1, 2]])) #Сортировка вероятности будет такая, как указано в clf.classes
#У predict_probа смотреть нужно на наименьшее число, так как это не вероятность, а кратчайшее расстояние до прямой. Predict =наименьшему числу.
'''
model = svm.SVC(probability=True)
model.fit(X, Y)
results = model.predict_proba(test_data)[0]

# gets a dictionary of {'class_name': probability}
prob_per_class_dictionary = dict(zip(model.classes_, results))

# gets a list of ['most_probable_class', 'second_most_probable_class', ..., 'least_class']
results_ordered_by_probability = map(lambda x: x[0], sorted(zip(model.classes_, results), key=lambda x: x[1], reverse=True))
'''
