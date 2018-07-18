import os
import re

from sklearn import metrics

# os.environ['KERAS_BACKEND'] = 'cntk'
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, Dropout, Concatenate, Bidirectional, LSTM, GRU
from keras.models import Model
from keras import backend as K
from keras.engine import Layer
from keras import initializers
from keras.models import model_from_json
import numpy

MAX_SEQUENCE_LENGTH = 1000
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2
BASE_PATH = "../data/"
MODEL_BASE_PATH = "../model/"


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


def train_convolution_model(x_train, y_train, x_val, y_val, word_index):
    epoch_size = 200
    embedding_matrix = numpy.random.random((len(word_index) + 1, EMBEDDING_DIM))
    embedding_layer = Embedding(len(word_index) + 1,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=True)
    convs = []
    filter_sizes = [3, 4, 5]

    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    for fsz in filter_sizes:
        l_conv = Conv1D(nb_filter=128, filter_length=fsz, activation='relu')(embedded_sequences)
        l_pool = MaxPooling1D(5)(l_conv)
        convs.append(l_pool)
    l_merge = Concatenate(axis=-1)(convs)
    l_cov1 = Conv1D(128, 5, activation='relu')(l_merge)
    l_pool1 = MaxPooling1D(5)(l_cov1)
    drop_1 = Dropout(0.25)(l_pool1)
    l_cov2 = Conv1D(128, 5, activation='relu')(drop_1)
    l_pool2 = MaxPooling1D(30)(l_cov2)
    drop_2 = Dropout(0.35)(l_pool2)
    l_flat = Flatten()(drop_2)
    l_dense = Dense(128, activation='relu')(l_flat)
    preds = Dense(2, activation='softmax')(l_dense)
    model = Model(sequence_input, preds)
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])
    model.fit(x_train, y_train, validation_data=(x_val, y_val), nb_epoch=epoch_size, batch_size=32)
    return model


def train_bidirectional_lstm(x_train, y_train, x_val, y_val, word_index):
    epoch_size = 150
    embedding_matrix = numpy.random.random((len(word_index) + 1, EMBEDDING_DIM))
    embedding_layer = Embedding(len(word_index) + 1,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=True)
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    l_lstm = Bidirectional(LSTM(100))(embedded_sequences)
    drop_1 = Dropout(0.20)(l_lstm)
    preds = Dense(2, activation='softmax')(drop_1)
    bi_lstm_model = Model(sequence_input, preds)
    bi_lstm_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])
    print("model fitting - Bidirectional LSTM")
    bi_lstm_model.summary()
    bi_lstm_model.fit(x_train, y_train, validation_data=(x_val, y_val), nb_epoch=epoch_size, batch_size=50)
    return bi_lstm_model


def train_bidirectional_lstm_han(x_train, y_train, x_val, y_val, word_index):
    class AttLayer(Layer):
        def __init__(self, **kwargs):
            self.init = initializers.get('normal')
            # self.input_spec = [InputSpec(ndim=3)]
            super(AttLayer, self).__init__(**kwargs)

        def build(self, input_shape):
            assert len(input_shape) == 3
            # self.W = self.init((input_shape[-1],1))
            self.W = self.init((input_shape[-1],))
            # self.input_spec = [InputSpec(shape=input_shape)]
            self.trainable_weights = [self.W]
            super(AttLayer, self).build(input_shape)  # be sure you call this somewhere!

        def call(self, x, mask=None):
            eij = K.tanh(K.dot(x, self.W))

            ai = K.exp(eij)
            weights = ai / K.sum(ai, axis=1).dimshuffle(0, 'x')

            weighted_input = x * weights.dimshuffle(0, 1, 'x')
            return weighted_input.sum(axis=1)

        def get_output_shape_for(self, input_shape):
            return (input_shape[0], input_shape[-1])

    epoch_size = 150
    embedding_matrix = numpy.random.random((len(word_index) + 1, EMBEDDING_DIM))
    embedding_layer = Embedding(len(word_index) + 1,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=True)
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    l_gru = Bidirectional(GRU(100, return_sequences=True))(embedded_sequences)
    drop_1 = Dropout(0.20)(l_gru)
    l_att = AttLayer()(drop_1)
    preds = Dense(2, activation='softmax')(l_att)
    bi_lstm_han_model = Model(sequence_input, preds)
    bi_lstm_han_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])
    print("model fitting - attention GRU network")
    bi_lstm_han_model.summary()
    bi_lstm_han_model.fit(x_train, y_train, validation_data=(x_val, y_val), nb_epoch=epoch_size, batch_size=50)
    return bi_lstm_han_model


def main():
    x_train, y_train = read_files("train")
    print("Training File Read - Size: ", len(x_train))
    x_test, y_test = read_files("test")
    print("Test File Read - Size: ", len(x_test))
    x_val, y_val = read_files("val")
    print("Validation File Read - Size: ", len(x_val))

    texts = x_train + x_val + x_test

    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(texts)
    train_sequences = tokenizer.texts_to_sequences(x_train)
    val_sequences = tokenizer.texts_to_sequences(x_val)
    test_sequences = tokenizer.texts_to_sequences(x_test)

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    x_train = pad_sequences(train_sequences, maxlen=MAX_SEQUENCE_LENGTH)
    x_val = pad_sequences(val_sequences, maxlen=MAX_SEQUENCE_LENGTH)
    x_test = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)

    y_train = to_categorical(numpy.asarray(y_train))
    y_val = to_categorical(numpy.asarray(y_val))
    y_test = to_categorical(numpy.asarray(y_test))

    print('Shape of data tensor:', x_train.shape + x_val.shape)
    print('Shape of label tensor:', y_train.shape + y_val.shape)

    print('Number of positive and negative points in traing and validation set ')
    print(y_train.sum(axis=0))
    print(y_val.sum(axis=0))
    '''
    conv_model = train_convolution_model(x_train, y_train, x_val, y_val, word_index)
    save_keras_model("convolution", conv_model)
    #conv_model = load_keras_model("convolution")
    accuracy, auc_score, precision_score = evaluate(conv_model, x_test, y_test)
    print "Convolution Model: Accuracy-{accuracy}, AUC-{auc_score} and PR_Score-{precision_score}" \
        .format(accuracy=accuracy, auc_score=auc_score, precision_score=precision_score)
    '''
    bi_lstm_model = train_bidirectional_lstm(x_train, y_train, x_val, y_val, word_index)
    save_keras_model("bi_lstm", bi_lstm_model)
    # bi_lstm_model = load_keras_model("bi_lstm")
    accuracy, auc_score, precision_score = evaluate(bi_lstm_model, x_test, y_test)
    print("Bidirectional LSTM Model: Accuracy-{accuracy}, AUC-{auc_score} and PR_Score-{precision_score}" \
          .format(accuracy=accuracy, auc_score=auc_score, precision_score=precision_score))


if __name__ == "__main__":
    main()
