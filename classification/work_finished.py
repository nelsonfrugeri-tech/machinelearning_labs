from pathlib import Path
from kit_learn import Classifier

import seaborn
import pandas
import matplotlib.pyplot as plt
import numpy
    
def scatterplot(axisX, axisY, hue, data):
    seaborn.scatterplot(x = axisX, y = axisY, hue = hue, data=data)
    plt.show()

def relplot(axisX, axisY, hue, col, data):
    seaborn.relplot(x = axisX, y = axisY, hue = hue, col = col, data=data)
    plt.show()

def main():
    classifier = Classifier()
    
    data = pandas.read_csv(Path(__file__).parent / "data/work_finished.csv")
    data["finished"] = data.unfinished.map({
        0: 1,
        1: 0
    })

    # classifier.baseline(data["finished"], 0.25)
    # classifier.learn(data[["expected_hours", "price"]], data["finished"], 20, 0.25)
    
    # scatterplot("expected_hours", "price", "finished", data)


    x_min = data[["expected_hours", "price"]][25:].min()
    x_max = data[["expected_hours", "price"]][25:].max()
    y_min = data["finished"][25:].min()
    y_max = data["finished"][25:].max()

    print(x_min, x_max, y_min, y_max)
    
    pixels = 100
    # eixo_x = numpy.arange(x_min, x_max, (x_max - x_min) / pixels)
    # eixo_y = numpy.arange(y_min, y_max, (y_max - y_min) / pixels)

    xx, yy = numpy.meshgrid(10, 10)
    pontos = numpy.c_[xx.ravel(), yy.ravel()]

    plt.contourf(xx, yy, pontos, alpha=0.3)

    # scatterplot("expected_hours", "price", data["finished"][25:], data[["expected_hours", "price"]][25:])

    # relplot("expected_hours", "price", "finished", "finished", data)

if __name__ == "__main__":
    main()