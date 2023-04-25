from pathlib import Path
from kit_learn import Classifier

import pandas

# features: home | how_it_works | contact
# results: bought = 0 not or 1 yes
mac = Classifier()
data = pandas.read_csv(Path(__file__).parent / "data/ecom.csv")

mac.linear_vector(data[["home", "how_it_works", "contact"]], data["bought"], 20, 0.25)
