from pathlib import Path
from sklearn.model_selection import train_test_split
from kit_learn import Machine

import seaborn
import pandas
import matplotlib.pyplot as plt

mac = Machine()

data = pandas.read_csv(Path(__file__).parent / "data/work_finished.csv")
data["finished"] = data.unfinished.map({
    0: 1,
    1: 0
})

features = data[["expected_hours", "price"]]
results = data["finished"]

train_features, test_features, train_results, test_results = train_test_split(features, results, 
                                                                              random_state = 20, test_size = 0.25,
                                                                              stratify = results)

# Training 75% of the data
mac.fit(train_features, train_results)

# Test with the remaining 25%
mac.hit_rate_in_percent(test_results, mac.predict(test_features))

seaborn.scatterplot(x = "expected_hours", y = "price", hue = "finished", data=data)
seaborn.relplot(x = "expected_hours", y = "price", hue = "finished", col = "finished", data=data)

plt.show()

