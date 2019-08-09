import pandas as pd 
import numpy as np 
import scipy.optimize as opt 
from sklearn import preprocessing 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, jaccard_score, classification_report, confusion_matrix, log_loss
import itertools 

# load data
churn_df = pd.read_csv("data/churn_data.csv")
churn_df = churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip', 'callcard', 'wireless','churn']]
churn_df['churn'] = churn_df['churn'].astype('int')

# extract data
X = np.asarray(churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip']])
y = np.asarray(churn_df['churn'])

# normalize dataset
X = preprocessing.StandardScaler().fit(X).transform(X)

# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# train model
LR = LogisticRegression(C=0.01, solver="liblinear").fit(X_train, y_train)

# predict 
yhat = LR.predict(X_test)
yhat_prob = LR.predict_proba(X_test)

# evaluate metrics 
def plot_confusion_matrix(cm, classes, normalize=False, title="Confusion matrix", cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix without normalization")
    
    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center",color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")

cnf_matrix = confusion_matrix(y_test, yhat, labels=[1, 0])
np.set_printoptions(precision=2)

print("Jaccard: {}".format(jaccard_score(y_test, yhat)))
print("Log loss: {}".format(log_loss(y_test, yhat_prob)))
print(classification_report(y_test, yhat))


plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['churn = 1', 'churn = 0'], normalize=False, title="Confusion matrix")
plt.show()