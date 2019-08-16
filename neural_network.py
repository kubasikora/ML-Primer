import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics

df = pd.read_csv('./data/diabetes.csv')
X = df[['PREGNANT_TIMES', 'PLASMA_GLUCOSE_CONCENTRATION', 'PRESSURE', 'SKIN_THICKNESS', 'INSULIN', 'BMI', 'PEDIGREE', 'AGE']].values
y = df['CLASS'].values

X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = Sequential()
model.add(Dense(64, use_bias=True, input_dim=8, activation='relu'))
model.add(Dense(32, use_bias=True, activation='relu'))
model.add(Dense(1, use_bias=True,activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

model.fit(X_train, y_train, epochs=150, batch_size=10)
_, accuracy = model.evaluate(X, y)

yhat = model.predict_classes(X_train)
metrics.accuracy_score(y_train, yhat)
print('Train set accuracy: {}'.format(accuracy*100))

yhat = model.predict_classes(X_test)
metrics.accuracy_score(y_test, yhat)
print('Test set accuracy: {}'.format(accuracy*100))