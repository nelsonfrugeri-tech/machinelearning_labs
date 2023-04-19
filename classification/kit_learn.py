from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

class Machine:
    def __init__(self):
        self.model = LinearSVC()

    def fit(self, features, results):
        self.model.fit(features, results)

    def predict(self, features):
        return self.model.predict(features)
    
    def hit_rate_in_percent(self, expected, predict):
        return accuracy_score(expected, predict) * 100
