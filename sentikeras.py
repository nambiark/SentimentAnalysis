import numpy as np
import pandas as pd
df = pd.read_pickle('atlastfinal.pkl')
#df['Results'] = (df['Precipitationmm'] > 0).astype(int)

result = df[['Compound', 'Confidence', 'Ensemble', 'Negative', 'Neutral','Positive', 'TextBlob','Sentiment']].as_matrix()
#result = df.as_matrix()
row = round(0.8 * result.shape[0])
train = result[:int(row), :]
import numpy as np
np.random.shuffle(train)
x_train = train[:, :-1]
y_train = train[:, -1]
x_test = result[int(row):, :-1]
y_test = result[int(row):, -1]

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD


model = Sequential()
model.add(Dense(64, input_dim=7, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='hard_sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='hard_sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='Adam',
              metrics=['accuracy'])
model.fit(x_train, y_train,
          epochs=100,
          batch_size=128)

score = model.evaluate(x_test, y_test, batch_size=128)
print score
