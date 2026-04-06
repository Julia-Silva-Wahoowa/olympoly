import pandas as pd
from olympoly.olympics_betting.regression_model.model import train_model


# load your dataset
df = pd.read_csv("your_dataset.csv")  # fix path

model = train_model(df)

print(model)

