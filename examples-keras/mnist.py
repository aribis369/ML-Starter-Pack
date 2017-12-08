from sklearn.preprocessing import normalize
from sklearn.metrics import make_scorer, f1_score, accuracy_score, recall_score, precision_score, classification_report, precision_recall_fscore_support
from sklearn.model_selection import cross_val_score, cross_val_predict
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, Model
from keras.layers import Embedding, Input, LSTM
from keras.layers import Activation, Dense, Dropout, Embedding, Flatten, Input, Merge, Conv2D, MaxPooling2D, GlobalMaxPooling2D, ZeroPadding2D, BatchNormalization
from keras.utils import np_utils
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.optimizers import Adam

batch_size=128
num_classes = 10
results = dict()

(X_train, y_train), (X_test, y_test) = mnist.load_data()

print(X_test)

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

Y_train = to_categorical(y_train, num_classes)
Y_test = to_categorical(y_test, num_classes)

def cnn_model():
    model = Sequential()

    model.add(Conv2D(32, (3, 3), input_shape=(28,28,1)))
    model.add(Activation('relu'))
    BatchNormalization(axis=-1)
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    BatchNormalization(axis=-1)
    model.add(Conv2D(64,(3, 3)))
    model.add(Activation('relu'))
    BatchNormalization(axis=-1)
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Flatten())
    # Fully connected layer

    BatchNormalization()
    model.add(Dense(512))
    model.add(Activation('relu'))
    BatchNormalization()
    model.add(Dropout(0.2))
    model.add(Dense(10))
    model.add(Activation('softmax'))

    return(model)

model=cnn_model()
print(model.summary())

model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

model.fit(X_train, Y_train, epochs=7, batch_size=batch_size)

y_pred=model.predict_on_batch(X_test)
y_pred = np.argmax(y_pred, axis=1)
Y_test = np.argmax(Y_test, axis=1)

print(classification_report(Y_test,y_pred))







