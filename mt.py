import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense
from tensorflow.keras.callbacks import ModelCheckpoint

# Load dataset
df = pd.read_csv('/content/Machinetransalation.csv', encoding='latin1')

# Add <sos> and <eos> to target (French) text
df['french'] = df['french'].apply(lambda x: '<sos> ' + x + ' <eos>')

# Prepare source and target
input_texts = df['english'].tolist()
target_texts = df['french'].tolist()

# Tokenization
input_tokenizer = Tokenizer()
target_tokenizer = Tokenizer(filters='')
input_tokenizer.fit_on_texts(input_texts)
target_tokenizer.fit_on_texts(target_texts)

input_seq = input_tokenizer.texts_to_sequences(input_texts)
target_seq = target_tokenizer.texts_to_sequences(target_texts)

max_encoder_seq_length = max(len(seq) for seq in input_seq)
max_decoder_seq_length = max(len(seq) for seq in target_seq)

input_seq = pad_sequences(input_seq, maxlen=max_encoder_seq_length, padding='post')
target_seq = pad_sequences(target_seq, maxlen=max_decoder_seq_length, padding='post')

num_encoder_tokens = len(input_tokenizer.word_index) + 1
num_decoder_tokens = len(target_tokenizer.word_index) + 1

# Encoder
encoder_inputs = Input(shape=(None,))
enc_emb = Embedding(num_encoder_tokens, 256)(encoder_inputs)
encoder_lstm, state_h, state_c = LSTM(256, return_state=True)(enc_emb)
encoder_states = [state_h, state_c]

# Decoder
decoder_inputs = Input(shape=(None,))
dec_emb_layer = Embedding(num_decoder_tokens, 256)
dec_emb = dec_emb_layer(decoder_inputs)
decoder_lstm = LSTM(256, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

# Prepare target data
target_seq_input = target_seq[:, :-1]
target_seq_output = target_seq[:, 1:]
target_seq_output = np.expand_dims(target_seq_output, -1)

# Model checkpoint
checkpoint = ModelCheckpoint('seq2seq_model.h5', monitor='loss', save_best_only=True, mode='min', verbose=1)

# Training
model.fit([input_seq, target_seq_input], target_seq_output, batch_size=32, epochs=100, callbacks=[checkpoint])

# Encoder Model
encoder_model = Model(encoder_inputs, encoder_states)

# Decoder Model
decoder_state_input_h = Input(shape=(256,))
decoder_state_input_c = Input(shape=(256,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
dec_emb2 = dec_emb_layer(decoder_inputs)
decoder_outputs2, state_h2, state_c2 = decoder_lstm(dec_emb2, initial_state=decoder_states_inputs)
decoder_states2 = [state_h2, state_c2]
decoder_outputs2 = decoder_dense(decoder_outputs2)
decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs2] + decoder_states2)

# Reverse target tokenizer for decoding
reverse_target_word_index = dict((i, word) for word, i in target_tokenizer.word_index.items())

def decode_sequence(input_seq):
    states_value = encoder_model.predict(input_seq)

    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = target_tokenizer.word_index['<sos>']

    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = reverse_target_word_index.get(sampled_token_index, '')

        if sampled_word == '<eos>' or len(decoded_sentence.split()) > max_decoder_seq_length:
            stop_condition = True
        else:
            decoded_sentence += ' ' + sampled_word
            target_seq = np.zeros((1, 1))
            target_seq[0, 0] = sampled_token_index
            states_value = [h, c]

    return decoded_sentence.strip()

# Test the model with a sentence
test_sentence = ['how are you']
test_seq = pad_sequences(input_tokenizer.texts_to_sequences(test_sentence), maxlen=max_encoder_seq_length, padding='post')
print(decode_sequence(test_seq))
