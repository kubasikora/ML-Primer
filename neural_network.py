import pandas as pd
from keras.models import Sequential
from keras.layers import Dense

df = pd.read_csv('./data/diabetes.csv')
X = df[['PREGNANT_TIMES', 'PLASMA_GLUCOSE_CONCENTRATION', 'PRESSURE', 'SKIN_THICKNESS', 'INSULIN', 'BMI', 'PEDIGREE', 'AGE']].values
y = df['CLASS'].values

model = Sequential()
model.add(Dense(64, use_bias=True, input_dim=8, activation='relu'))
model.add(Dense(32, use_bias=True, activation='relu'))
model.add(Dense(1, use_bias=True,activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

model.fit(X, y, epochs=150, batch_size=10)
_, accuracy = model.evaluate(X, y)
print('Accuracy: {}'.format(accuracy*100))
