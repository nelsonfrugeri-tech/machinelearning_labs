from pathlib import Path
from kit_learn import Machine

import seaborn
import pandas
import matplotlib.pyplot as plt
    
def dashboards(data):
    seaborn.scatterplot(x = "expected_hours", y = "price", hue = "finished", data=data)
    seaborn.relplot(x = "expected_hours", y = "price", hue = "finished", col = "finished", data=data)

    plt.show()

def main():
    mac = Machine()
    data = pandas.read_csv(Path(__file__).parent / "data/work_finished.csv")
    data["finished"] = data.unfinished.map({
        0: 1,
        1: 0
    })

    mac.baseline(data["finished"], 0.25)
    mac.classification(data[["expected_hours", "price"]], data["finished"], 20, 0.25)
    # dashboards(data)

if __name__ == "__main__":
    main()