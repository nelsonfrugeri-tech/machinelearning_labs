from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import numpy

class Classifier:
    def __init__(self):
        self.model = LinearSVC()

    def learn(self, features, results, random_state, test_size):
        train_features, test_features, train_results, test_results = train_test_split(features, results, 
                                                                                random_state = random_state, 
                                                                                test_size = test_size,
                                                                                stratify = results)

        self.__fit(train_features, train_results)
        self.__hit_rate_in_percent(test_results, self.__predict(test_features))
        self.__proportion_train_test(train_results, test_results)

    def baseline(self, results, test_size):
        tsize = int(test_size * 100)
        print("Baseline has a hit rate: {} \n".format(accuracy_score(results[tsize:], numpy.ones(len(results[tsize:]))) * 100))

    def __fit(self, features, results):
        print("{} elements for the training \n".format(len(features)))
        self.model.fit(features, results)

    def __predict(self, features):
        print("{} elements for the testing \n".format(len(features)))
        return self.model.predict(features)
    
    def __hit_rate_in_percent(self, expected, predict):
        print("Hit rate: {} \n".format(accuracy_score(expected, predict) * 100))

    def __proportion_train_test(self, train, test):
        print("Proportion between train and test is: \n")
        print(train.value_counts(), "\n")
        print(test.value_counts(), "\n")
