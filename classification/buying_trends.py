from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import pandas
from pathlib import Path

# features: home | how_it_works | contact
# results: bought = 0 not or 1 yes

data = pandas.read_csv(Path(__file__).parent / "data/ecom.csv")

features = data[["home", "how_it_works", "contact"]]
results = data["bought"]

train_features, test_features, train_results, test_results = train_test_split(features, results, random_state = 20, test_size = 0.25)

# Training 75% of the data
model = LinearSVC()
model.fit(train_features, train_results)

# Test with the remaining 25%
print("Hit rate: %.2f" % (accuracy_score(test_results, model.predict(test_features)) * 100))
