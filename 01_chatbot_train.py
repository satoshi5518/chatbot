import pickle
import re
import numpy as np
from keras.models import Model
from keras.layers import Dense, GRU, Input, Masking
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping 

with open('chars.pickle', mode='rb') as f:
    chars_list = pickle.load(f)
#print(chars_list)


# インデックスと文字で辞書を作成
char_indices = {}  # 文字がキーでインデックスが値
for i, char in enumerate(chars_list):
    char_indices[char] = i
indices_char = {}  # インデックスがキーで文字が値
for i, char in enumerate(chars_list):
    indices_char[i] = char

with open(".\\text\\question.txt", mode="r", encoding="shift_jis") as f:  # 前回保存したファイル
    q_text = f.read()
#print(q_text)
seperator = "@"
q_sentence_list = q_text.split(seperator) 
q_sentence_list.pop() 
q_sentence_list = [re.sub("[\n\t]", "", x) for x in q_sentence_list]
#print(q_sentence_list)

with open(".\\text\\answer.txt", mode="r", encoding="shift_jis") as f:  # 前回保存したファイル
    a_text = f.read()
#print(a_text)
a_sentence_list = a_text.split(seperator) 
a_sentence_list.pop() 
a_sentence_list = [re.sub("[\n\t]", "", x) for x in a_sentence_list]
#print(a_sentence_list)

max_sentence_length = 15  # 文章の最大長さ。これより長い文章はカットされる。
q_sentence_list = [sentence for sentence in q_sentence_list if len(sentence) <= max_sentence_length]  # 長すぎる文章のカット
a_sentence_list = [sentence for sentence in a_sentence_list if len(sentence) <= max_sentence_length]  # 長すぎる文章のカット

n_char = len(chars_list)  # 文字の種類の数
q_n_sample = len(q_sentence_list)   # サンプル数
a_n_sample = len(a_sentence_list)   # サンプル数
n_sample = q_n_sample 
#print(n_char)
#print(q_n_sample)
#print(a_n_sample)

x_sentences = []  # 入力の文章
t_sentences = []  # 正解の文章
for i in range(q_n_sample):
    x_sentences.append(q_sentence_list[i])
for i in range(a_n_sample):
    t_sentences.append("\t" + a_sentence_list[i] + "\n") 
max_length_x = max_sentence_length  # 入力文章の最大長さ
max_length_t = max_sentence_length + 2  # 正解文章の最大長さ
#print(x_sentences)
#print(t_sentences)

x_encoder = np.zeros((n_sample, max_length_x, n_char), dtype=np.bool)  # encoderへの入力
x_decoder = np.zeros((n_sample, max_length_t, n_char), dtype=np.bool)  # decoderへの入力
t_decoder = np.zeros((n_sample, max_length_t, n_char), dtype=np.bool)  # decoderの正解

for i in range(n_sample):
    x_sentence = x_sentences[i]
    t_sentence = t_sentences[i]
    for j, char in enumerate(x_sentence):
        x_encoder[i, j, char_indices[char]] = 1  # encoderへの入力をone-hot表現で表す
    for j, char in enumerate(t_sentence):
        x_decoder[i, j, char_indices[char]] = 1  # decoderへの入力をone-hot表現で表す
        if j > 0:  # 正解は入力より1つ前の時刻のものにする
            t_decoder[i, j-1, char_indices[char]] = 1
            
#print(x_encoder.shape)

#Model構築
batch_size = 32
epochs = 3000
n_mid = 256  # 中間層のニューロン数

encoder_input = Input(shape=(None, n_char))
encoder_mask = Masking(mask_value=0)  # 全ての要素が0であるベクトルの入力は無視する
encoder_masked = encoder_mask(encoder_input)
encoder_lstm = GRU(n_mid, dropout=0.2, recurrent_dropout=0.2, return_state=True)  # dropoutを設定し、ニューロンをランダムに無効にする
encoder_output, encoder_state_h = encoder_lstm(encoder_masked)

decoder_input = Input(shape=(None, n_char))
decoder_mask = Masking(mask_value=0)  # 全ての要素が0であるベクトルの入力は無視する
decoder_masked = decoder_mask(decoder_input)
decoder_lstm = GRU(n_mid, dropout=0.2, recurrent_dropout=0.2, return_sequences=True, return_state=True)  # dropoutを設定
decoder_output, _ = decoder_lstm(decoder_masked, initial_state=encoder_state_h)  # encoderの状態を初期状態にする
decoder_dense = Dense(n_char, activation='softmax')
decoder_output = decoder_dense(decoder_output)

model = Model([encoder_input, decoder_input], decoder_output)

model.compile(loss="categorical_crossentropy", optimizer="rmsprop")
print(model.summary())

# val_lossに改善が見られなくなってから、30エポックで学習は終了
early_stopping = EarlyStopping(monitor="val_loss", patience=30) 

history = model.fit([x_encoder, x_decoder], t_decoder,
                     batch_size=batch_size,
                     epochs=epochs,
                     validation_split=0.1,  # 10%は検証用
                     callbacks=[early_stopping])


loss = history.history['loss']
val_loss = history.history['val_loss']

plt.plot(np.arange(len(loss)), loss)
plt.plot(np.arange(len(val_loss)), val_loss)
plt.show()

# encoderのモデル
encoder_model = Model(encoder_input, encoder_state_h)

# decoderのモデル
decoder_state_in_h = Input(shape=(n_mid,))
decoder_state_in = [decoder_state_in_h]

decoder_output, decoder_state_h = decoder_lstm(decoder_input,
                                               initial_state=decoder_state_in_h)
decoder_output = decoder_dense(decoder_output)

decoder_model = Model([decoder_input] + decoder_state_in,
                      [decoder_output, decoder_state_h])

# モデルの保存
encoder_model.save('encoder_model.h5')
decoder_model.save('decoder_model.h5')
