import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
import tensorflow.keras.backend as K

from tensorflow.keras.utils import to_categorical

from tensorflow.keras.models import model_from_json

##############################
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import argparse
##############################

df_train = pd.read_csv('model/train_data.csv', index_col = False)
labels = df_train[['784']]
df_train.head()

df_train.drop(df_train.columns[[784]], axis=1, inplace=True)
np.random.seed(1212)


labels = np.array(labels)

categorical_data = to_categorical(labels, num_classes = 13)

l = []
for i in range(df_train.shape[0]):
    l.append(np.array(df_train[i:i+1]).reshape(28, 28, 1))
    
print(len(l))    
train_X, test_X, train_y, test_y = train_test_split(np.array(l), categorical_data, test_size=0.20, random_state=42)

print(len(train_X))
print(len(train_y))
print(len(test_X))

np.random.seed(7)


model = keras.Sequential([
    layers.Conv2D(30, (5, 5), input_shape=(28, 28, 1), activation='relu'), 
    layers.MaxPool2D(pool_size=2), 
    layers.Conv2D(15, (3, 3), activation='relu'), 
    layers.MaxPool2D(pool_size=2), 
    layers.Dropout(0.2), 
    layers.Flatten(), 
    layers.Dense(128, activation='relu'), 
    layers.Dense(50, activation='relu'), 
    layers.Dense(13, activation='softmax'), 
])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#from tensorflow.keras.models import model_from_json

e=20

H=model.fit(train_X, train_y, validation_split=0.25, epochs=e, batch_size=200, shuffle=True, verbose=1)

###################################################
predictions = model.predict(test_X)
# 將預測值轉換為類別
predicted_labels = np.argmax(predictions, axis=1)
# 打印分類報告
print(classification_report(np.argmax(test_y, axis=1), predicted_labels))
##################################################




model_json = model.to_json()
with open('model/model.json', 'w') as json_file:
    json_file.write(model_json)
model.save_weights('model/model_weights.weights.h5')


# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, e), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, e), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, e), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, e), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
#plt.savefig(args["output"])
plt.show()


