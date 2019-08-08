import numpy as np 
import pandas as pd 
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
from six import StringIO
import pydotplus
import matplotlib.image as mpimg
from sklearn import tree

# load data
df = pd.read_csv('./data/drugs.csv')

# preprocess data
X = df[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values # remove headers
le_sex = preprocessing.LabelEncoder()
le_sex.fit(['F', 'M'])
X[:,1] = le_sex.transform(X[:,1])

le_BP = preprocessing.LabelEncoder()
le_BP.fit(['LOW', 'NORMAL', 'HIGH'])
X[:,2] = le_BP.transform(X[:,2])

le_Chol = preprocessing.LabelEncoder()
le_Chol.fit(['NORMAL', 'HIGH'])
X[:,3] = le_Chol.transform(X[:,3]) 

y = df['Drug']

# split data
X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.3)

# create classifier
drugTree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
drugTree.fit(X_trainset, y_trainset)

# predict 
predictions = drugTree.predict(X_testset)

# eval metrics
print("Decision trees's Accuracy: ", metrics.accuracy_score(y_testset, predictions))

# visualize the tree
dot_data = StringIO()
filename = "drugtree.png"
featureNames = df.columns[0:5]
targetNames = df["Drug"].unique().tolist()
out=tree.export_graphviz(drugTree,feature_names=featureNames, out_file=dot_data, class_names= np.unique(y_trainset), filled=True,  special_characters=True,rotate=False)  
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png(filename)
img = mpimg.imread(filename)
plt.figure(figsize=(100, 200))
plt.imshow(img,interpolation='nearest')