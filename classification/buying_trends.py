from pathlib import Path
from kit_learn import Machine

import pandas

# features: home | how_it_works | contact
# results: bought = 0 not or 1 yes
mac = Machine()
data = pandas.read_csv(Path(__file__).parent / "data/ecom.csv")

mac.classification(data[["home", "how_it_works", "contact"]], data["bought"], 20, 0.25)
