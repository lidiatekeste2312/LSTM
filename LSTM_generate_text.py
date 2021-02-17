import numpy as np
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import text_to_word_sequence
from tensorflow.keras.preprocessing.text import Tokenizer as tk
from tensorflow.keras.backend import one_hot
import tensorflow.keras.utils as np_utils
import matplotlib.pyplot as plt
raw_txt = open("Jolene.txt").read()

raw_txt_sequence = text_to_word_sequence(raw_txt)

tk = tk(split=' ')
tk.fit_on_texts([raw_txt_sequence])

lines = raw_txt.split("\n")
sequences = tk.texts_to_sequences(lines)
ngrams = []
max_len = 0
for l in sequences:
    for i in range(len(l)):
        ngrams.append(l[:i+1])
    if max_len < len(l):
        max_len = len(l)
x = [ngram[:-1] for ngram in ngrams]
y = [ngram[-1] for ngram in ngrams]

x_pad = []
for ip_x in x:
    num_padding = max_len - len(ip_x)
    x_pad.append(([0]*num_padding) + ip_x)
    
x = np.array(x_pad)
x = x.reshape(17750,1,308)
x = x.astype("float32")
y = np.array(y)
#y = np_utils.to_categorical(y)
num_classes = len(tk.word_index)+1
#x = one_hot(x, num_classes)
#y = one_hot(y, num_classes)
model = Sequential()
model.add(layers.LSTM(128))
model.add(layers.Dense(num_classes, activation = "softmax"))
model.compile(loss = "sparse_categorical_crossentropy", optimizer = "adam", metrics = ['accuracy'])
history = model.fit(x, y, batch_size = 64, epochs = 10)
# b = 0
# for i in range(1770,(len(x)-50),1770):
#     x_batch = one_hot(x[b:i], num_classes)
#     model.fit(x_batch, y[b:i], batch_size = 64, epochs = 1)
#     b = i

NUM_LINES = 1
for i in range(NUM_LINES):
    generated_sequence = np.zeros((max_len,))
   # generated_sequence = generated_sequence.reshape(1,308)
    for word in range(max_len):
        #ip_one_hot = one_hot(generated_sequence, num_classes)
        generated_sequence = generated_sequence.reshape(1,308)
        prediction = model.predict(
            generated_sequence[None], verbose = 0)[0]
        sampled_token = np.random.choice(
            np.arange(num_classes), p=prediction)
        #print(generated_sequence)
        generated_sequence = np.append(
            generated_sequence[0,1:],sampled_token)
        print(generated_sequence)
        #print(generated_sequence)
        #generated_sequence = generated_sequence.reshape(1,308)
        #generated_sequence = generated_sequence.astype(int)
    generated_txt = tk.sequences_to_texts([generated_sequence])[0]
    print("Sample {}: {}".format(i, generated_txt))

plt.plot(history.history['accuracy'])

plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train accuracy'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])

plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train loss'], loc='upper left')
plt.show()