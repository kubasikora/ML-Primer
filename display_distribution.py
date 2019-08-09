import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 

cell_df = pd.read_csv("data/cell_samples.csv")
cell_df = cell_df[pd.to_numeric(cell_df['BareNuc'], errors="coerce").notnull()]
cell_df['BareNuc'] = cell_df['BareNuc'].astype('int')
cell_df['Class'] = cell_df['Class'].astype('int')

ax = cell_df[cell_df['Class'] == 4][:].plot(kind='scatter', x='Clump', y='UnifSize', color='DarkBlue', label='malignant');
cell_df[cell_df['Class'] == 2][:].plot(kind='scatter', x='Clump', y='UnifSize', color='Yellow', label='benign', ax=ax);
plt.show()