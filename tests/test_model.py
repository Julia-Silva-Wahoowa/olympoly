import pandas as pd
from olympoly.load_data import load_olympic_data
from olympoly.olympics_betting.regression_model.model import train_model


# load your datase
df = load_olympic_data()
model = train_model(df)

print(model)
