import pandas as pd 
import numpy as np
import scipy.optimize as opt 
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 
from sklearn import svm
import itertools
from sklearn.metrics import confusion_matrix, jaccard_score, classification_report, confusion_matrix, f1_score

cell_df = pd.read_csv("data/cell_samples.csv")
cell_df = cell_df[pd.to_numeric(cell_df['BareNuc'], errors="coerce").notnull()]
cell_df['BareNuc'] = cell_df['BareNuc'].astype('int')
cell_df['Class'] = cell_df['Class'].astype('int')

feature_df = cell_df[['Clump', 'UnifSize', 'UnifShape', 'MargAdh', 'SingEpiSize', 'BareNuc', 'BlandChrom', 'NormNucl', 'Mit']]
X = np.asarray(feature_df)

y = np.asarray(cell_df['Class'])
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)

clf = svm.SVC(kernel='rbf', gamma="auto")
clf.fit(X_train, y_train) 
yhat = clf.predict(X_test)

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

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
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

cnf_matrix = confusion_matrix(y_test, yhat, labels=[2,4])
np.set_printoptions(precision=2)

y_test_bin = list(map(lambda x: 1 if x == 2 else 0, y_test))
y_hat_bin = list(map(lambda x: 1 if x == 2 else 0, yhat))

print("F1 score: {}".format(f1_score(y_test, yhat, average='weighted')))
print("Jaccard: {}".format(jaccard_score(y_test_bin, y_hat_bin)))
print(classification_report(y_test, yhat))

plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['Benign(2)','Malignant(4)'],normalize= False,  title='Confusion matrix')
plt.show()