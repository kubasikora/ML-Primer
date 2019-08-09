import pandas as pd 
import matplotlib.pyplot as plt

df = pd.read_csv("./data/tele_cust.csv")
viz = df[['tenure','age','marital','income']]
viz.hist()
plt.show()