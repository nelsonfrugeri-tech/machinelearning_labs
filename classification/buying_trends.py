from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import pandas
from pathlib import Path

# features: home | how_it_works | contact
# results: bought = 0 not or 1 yes

data = pandas.read_csv(Path(__file__).parent / "data/ecom.csv")

features = data[["home", "how_it_works", "contact"]]
results = data["bought"]

# Training 75% of the data
model = LinearSVC()
model.fit(features[:75], results[:75])

# Test with the remaining 25%
print("Hit rate: %.2f" % (accuracy_score(results[75:], model.predict(features[75:])) * 100))
