from sklearn.model_selection import train_test_split
from pathlib import Path
from kit_learn import Machine

import pandas

# features: home | how_it_works | contact
# results: bought = 0 not or 1 yes
mac = Machine()
data = pandas.read_csv(Path(__file__).parent / "data/ecom.csv")

features = data[["home", "how_it_works", "contact"]]
results = data["bought"]

train_features, test_features, train_results, test_results = train_test_split(features, results, 
                                                                              random_state = 20, test_size = 0.25,
                                                                              stratify = results)

# Training 75% of the data
mac.fit(train_features, train_results)

# Test with the remaining 25%
print("Hit rate: %.2f \n" % (mac.hit_rate_in_percent(test_results, mac.predict(test_features))))

# Proportion between train and test
print("Proportion between train and test is: \n")
print(train_results.value_counts(), "\n")
print(test_results.value_counts(), "\n")
