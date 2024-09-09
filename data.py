import pandas as pd

df = pd.read_csv("altura_peso.csv")

x = df["Height"].values
y = df["Weight"].values


