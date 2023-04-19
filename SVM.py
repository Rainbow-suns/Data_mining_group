import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from data_loader import label_image
from sift_extractors import create_feature_bow, extract_sift_features, kmean_bow


class SVM:

    def __init__(self, kernel='linear', C=10000.0, max_iter=100000, degree=3, gamma=1):
        self.kernel = {'poly': lambda x, y: np.dot(x, y.T) ** degree,
                       'rbf': lambda x, y: np.exp(-gamma * np.sum((y - x[:, np.newaxis]) ** 2, axis=-1)),
                       'linear': lambda x, y: np.dot(x, y.T)}[kernel]
        self.C = C
        self.max_iter = max_iter

    def restrict_to_square(self, t, v0, u):
        t = (np.clip(v0 + t * u, 0, self.C) - v0)[1] / u[1]
        return (np.clip(v0 + t * u, 0, self.C) - v0)[0] / u[0]

    def fit(self, X, y):
        self.X = X.copy()
        self.y = y * 2 - 1
        self.lambdas = np.zeros_like(self.y, dtype=float)
        self.K = self.kernel(self.X, self.X) * self.y[:, np.newaxis] * self.y

        for _ in range(self.max_iter):
            for idxM in range(len(self.lambdas)):
                idxL = np.random.randint(0, len(self.lambdas))
                Q = self.K[[[idxM, idxM], [idxL, idxL]], [[idxM, idxL], [idxM, idxL]]]
                v0 = self.lambdas[[idxM, idxL]]
                k0 = 1 - np.sum(self.lambdas * self.K[[idxM, idxL]], axis=1)
                u = np.array([-self.y[idxL], self.y[idxM]])
                t_max = np.dot(k0, u) / (np.dot(np.dot(Q, u), u) + 1E-15)
                self.lambdas[[idxM, idxL]] = v0 + u * self.restrict_to_square(t_max, v0, u)

        idx, = np.nonzero(self.lambdas > 1E-15)
        self.b = np.mean((1.0 - np.sum(self.K[idx] * self.lambdas, axis=1)) * self.y[idx])

    def decision_function(self, X):
        return np.sum(self.kernel(X, self.X) * self.y * self.lambdas, axis=1) + self.b

    def predict(self, X):
        return (np.sign(self.decision_function(X)) + 1) // 2


if __name__ == '__main__':
    path = './data/raw_data/train/'
    startTime = datetime.datetime.now()
    data_train, label, label2id = label_image(path)
    print('Size of traing data set：', len(data_train))
    print('Size of labels', len(label))
    print('The relationships between labels and numbers', label2id)
    image_desctiptors = extract_sift_features(data_train)
    all_descriptors = []
    for descriptor in image_desctiptors:
        if descriptor is not None:
            for des in descriptor:
                all_descriptors.append(des)
    num_cluster = 60
    BoW = kmean_bow(all_descriptors, num_cluster)
    X_features = create_feature_bow(image_desctiptors, BoW, num_cluster)
    label = np.array(label)
    X_features = np.array(X_features)
    model_svm = SVM(kernel='rbf', C=1000, max_iter=100, gamma=0.001)
    model_svm.fit(X_features, label)
    endTime = datetime.datetime.now()
    print("SVM training time: " + str(endTime - startTime) + ".")
    test_path = './data/raw_data/test'
    data_test, test_label, test_label2id = label_image(test_path)
    print('Size of test data set：', len(data_test))
    print('Size of labels', len(test_label))
    print('The relationships between labels and numbers', test_label2id)
    test_image_desctiptors = extract_sift_features(data_test)
    test_all_descriptors = []
    for descriptor in test_image_desctiptors:
        if descriptor is not None:
            for des in descriptor:
                test_all_descriptors.append(des)
    test_features = create_feature_bow(test_image_desctiptors, BoW, num_cluster)
    test_features = np.array(test_features)
    test_label = np.array(test_label)
    y_test_pred = model_svm.predict(test_features)
    print('Model accuracy score: {0:0.3f}'.format(accuracy_score(test_label, y_test_pred)))
    cm = confusion_matrix(test_label, y_test_pred)
    cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive:1', 'Actual Negative:0'],
                             index=['Predict Positive:1', 'Predict Negative:0'])
    sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')
    plt.show()
