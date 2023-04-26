from pathlib import Path
from datetime import datetime
from kit_learn import Classifier

import numpy as np
import pandas as pd

SEED = 20
SIZE = 0.25
MAX_DEPTH = 3
N_SPLIT = 5

def buildData():
    np.random.seed(SEED)
    data = pd.read_csv(Path(__file__).parent / "data/garage_cars.csv")

    data.sold = data.sold.map({
        'no': 0,
        'yes': 1
    })

    data["model_age"] = datetime.today().year - data.model_year
    data["km_per_age"] = data.mileage_per_year * 1.60934
    data["model"] = data.model_age + np.random.randint(-2, 3, size = 10000)
    data.model = data.model + abs(data.model.min())

    data = data.drop(columns = ["Unnamed: 0", "mileage_per_year", "model_year"], axis = 1)

    return data

def main():
    data = buildData()
    mac = Classifier(SEED)

    mac.linear_vector(data[["price", "km_per_age", "model_age"]], data["sold"], SEED, SIZE)
    
    mac.baseline_by_dummy(data[["price", "km_per_age", "model_age"]], data["sold"], SEED, SIZE)

    mac.vector(data[["price", "km_per_age", "model_age"]], data["sold"], SEED, SIZE)
    
    mac.tree_plot(data[["price", "km_per_age", "model_age"]], 
                             data["sold"], SEED, SIZE, MAX_DEPTH)

    mac.tree_cross(data[["price", "km_per_age", "model_age"]], 
                             data["sold"], MAX_DEPTH, N_SPLIT)

    mac.tree_cross_groups(data[["price", "km_per_age", "model_age"]], 
                             data["sold"], data["model"], MAX_DEPTH, N_SPLIT)
    
    mac.pipeline(data[["price", "km_per_age", "model_age"]], 
                             data["sold"], data["model"], N_SPLIT)

if __name__ == "__main__":
    main()