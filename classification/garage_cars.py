from pathlib import Path
from datetime import datetime
from kit_learn import Classifier

import pandas as pd

SEED = 20
SIZE = 0.25

def buildData():
    data = pd.read_csv(Path(__file__).parent / "data/garage_cars.csv")

    data.sold = data.sold.map({
        'no': 0,
        'yes': 1
    })

    data["model_age"] = datetime.today().year - data.model_year
    data["km_per_age"] = data.mileage_per_year * 1.60934

    data = data.drop(columns = ["Unnamed: 0", "mileage_per_year", "model_year"], axis = 1)

    return data

def main():
    data = buildData()
    mac = Classifier(SEED)

    mac.linear_vector(data[["price", "km_per_age", "model_age"]], data["sold"], SEED, SIZE)
    mac.baseline_by_dummy(data[["price", "km_per_age", "model_age"]], data["sold"], SEED, SIZE)

    mac.vector(data[["price", "km_per_age", "model_age"]], data["sold"], SEED, SIZE)

if __name__ == "__main__":
    main()