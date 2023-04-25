from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler

import numpy

class Classifier:
    def __init__(self, random_state):
        self.model = LinearSVC(random_state=random_state)
        self.model_turbo = SVC(random_state=random_state)

    def vector(self, features, results, random_state, test_size):
        raw_train_features, raw_test_features, train_results, test_results = train_test_split(features, results, 
                                                                        random_state = random_state, 
                                                                        test_size = test_size,
                                                                        stratify = results)
        scaler = StandardScaler()
        scaler.fit(raw_train_features)
        train_features = scaler.transform(raw_train_features)
        test_features = scaler.transform(raw_test_features)

        self.model_turbo.fit(train_features, train_results)
        self.__print_hit_rate_in_percent(test_results, self.model_turbo.predict(test_features))
        self.__print__proportion_train_test(train_results, test_results)

    def linear_vector(self, features, results, random_state, test_size):
        train_features, test_features, train_results, test_results = train_test_split(features, results, 
                                                                                random_state = random_state, 
                                                                                test_size = test_size,
                                                                                stratify = results)

        self.model.fit(train_features, train_results)
        self.__print_hit_rate_in_percent(test_results, self.model.predict(test_features))
        self.__print__proportion_train_test(train_results, test_results)

    def baseline_by_dummy(self, features, results, random_state, test_size):
        dummy = DummyClassifier()
        train_features, test_features, train_results, test_results = train_test_split(features, results, 
                                                                        random_state = random_state, 
                                                                        test_size = test_size,
                                                                        stratify = results)
        
        dummy.fit(train_features, train_results)
        self.__print_hit_rate_in_percent(test_results, dummy.predict(test_features))
        self.__print__proportion_train_test(train_results, test_results)


    def baseline(self, results, test_size):
        tsize = int(test_size * 100)
        print("Baseline has a hit rate: {} \n".format(accuracy_score(results[tsize:], numpy.ones(len(results[tsize:]))) * 100))
    
    def __print_hit_rate_in_percent(self, expected, predict):
        print("Hit rate: {} \n".format(accuracy_score(expected, predict) * 100))

    def __print__proportion_train_test(self, train, test):
        print("Proportion between train and test is: \n")
        print(train.value_counts(), "\n")
        print(test.value_counts(), "\n")
