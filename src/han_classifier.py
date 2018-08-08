import re
import numpy
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn import metrics

from src.AttentionWithContext import Attention


from keras.models import Model, model_from_json
from keras.layers import Dense, Embedding, Input
from keras.layers import LSTM, Bidirectional, Dropout


BASE_PATH = "../data/"
MODEL_BASE_PATH = "../model/"


def calculate_accuracy(model, x_test, y_test, threshold):
    pred = model.predict(x_test)
    cnt = 0
    for i in range(0, len(pred)):
        if pred[i][0] >= threshold and y_test[i][0] == 1:
            cnt += 1
        elif pred[i][1] >= threshold and y_test[i][1] == 1:
            cnt += 1
    return float(cnt) * 100.0 / len(pred)


def evaluate(model, x_test, y_true):
    y_pred = model.predict(x_test)
    accuracy = calculate_accuracy(model, x_test, y_true, 0.5)
    y_actual = []
    for item in y_true:
        if item[0] == 1:
            y_actual.append(0)
        else:
            y_actual.append(1)
    auc_score = metrics.roc_auc_score(y_true, y_pred)
    precision_score = metrics.average_precision_score(y_true, y_pred)
    return accuracy, auc_score, precision_score


def save_keras_model(name, model):
    model_json = model.to_json()
    with open(MODEL_BASE_PATH + name + ".json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights(MODEL_BASE_PATH + name + ".h5")
    print("Saved model to disk")


def load_keras_model(name):
    json_file = open(MODEL_BASE_PATH + name + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(MODEL_BASE_PATH + name + ".h5")
    print("Loaded model from disk")
    return loaded_model


def remove_hyperlink(text):
    text = re.sub(r"http\S+", '', text.decode("utf-8"))
    return text


def read_files(file_type):
    x_list = []
    y_list = []
    with open(BASE_PATH + file_type + "_x.txt", "rb") as f:
        for line in f:
            processed_line = remove_hyperlink(line)
            x_list.append(processed_line.strip())

    with open(BASE_PATH + file_type + "_y.txt", "rb") as f:
        for line in f:
            y_list.append(int(line.strip()))

    return x_list, y_list


def clean_str(string):
    """
    Tokenization/string cleaning for dataset
    Every dataset is lower cased except
    """
    string = re.sub(r"\\", "", string)
    string = re.sub(r"\'", "", string)
    string = re.sub(r"\"", "", string)
    return string.strip().lower()


def BidLstm(maxlen, max_features, embed_size, embedding_matrix):
    inp = Input(shape=(maxlen, ))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix],
                  trainable=False)(inp)
    x = Bidirectional(LSTM(150, return_sequences=True, dropout=0.15,
                           recurrent_dropout=0.15))(x)
    x = Attention(maxlen)(x)
    x = Dense(100, activation="relu")(x)
    x = Dropout(0.15)(x)
    x = Dense(2, activation="softmax")(x)
    model = Model(inputs=inp, outputs=x)
    return model


x_train, y_train = read_files("train")
print("Training File Read - Size: ", len(x_train))
x_test, y_test = read_files("test")
print("Test File Read - Size: ", len(x_test))
x_val, y_val = read_files("val")
print("Validation File Read - Size: ", len(x_val))

texts = x_train + x_val + x_test
max_features = 20000
maxlen = 500
embed_size = 100

tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(texts)
train_sequences = tokenizer.texts_to_sequences(x_train)
val_sequences = tokenizer.texts_to_sequences(x_val)
test_sequences = tokenizer.texts_to_sequences(x_test)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

x_train = pad_sequences(train_sequences, maxlen=maxlen)
x_val = pad_sequences(val_sequences, maxlen=maxlen)
x_test = pad_sequences(test_sequences, maxlen=maxlen)

y_train = to_categorical(numpy.asarray(y_train))
y_val = to_categorical(numpy.asarray(y_val))
y_test = to_categorical(numpy.asarray(y_test))
nb_words = min(max_features, len(word_index))
embedding_matrix = numpy.zeros((nb_words, embed_size))
han_model = BidLstm(maxlen, max_features, embed_size, embedding_matrix)
han_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
han_model.fit(x_train, y_train, batch_size=25, epochs=10
              , validation_data=(x_val, y_val))
save_keras_model("han", han_model)
# bi_lstm_model = load_keras_model("han")
accuracy, auc_score, precision_score = evaluate(han_model, x_test, y_test)
print("HAN Model: Accuracy-{accuracy}, AUC-{auc_score} and PR_Score-{precision_score}"
      .format(accuracy=accuracy, auc_score=auc_score, precision_score=precision_score))
