import sys
from crowdkit.aggregation import DawidSkene
from crowdkit.datasets import load_dataset

import pandas as pd

df = pd.read_csv(sys.argv[1])  # should contain columns: worker, task, label
aggregated_labels = DawidSkene(n_iter=50).fit_predict(df)
for item in aggregated_labels.to_dict().items():
    print(item[0], item[1])