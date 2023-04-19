from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

class Machine:
    def __init__(self):
        self.model = LinearSVC()

    def fit(self, features, results):
        print("{} elements for the training \n".format(len(features)))
        self.model.fit(features, results)

    def predict(self, features):
        print("{} elements for the testing \n".format(len(features)))
        return self.model.predict(features)
    
    def hit_rate_in_percent(self, expected, predict):
        print("Hit rate: {} \n".format(accuracy_score(expected, predict) * 100))
