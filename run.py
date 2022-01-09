import util 
import models
import collections

"""
### Data processing
"""

# Load English data
english_sentences = util.load_data('data/small_vocab_en')
# Load French data
french_sentences = util.load_data('data/small_vocab_fr')

english_words_counter = collections.Counter([word for sentence in english_sentences for word in sentence.split()])
french_words_counter = collections.Counter([word for sentence in french_sentences for word in sentence.split()])

preproc_english_sentences, preproc_french_sentences, english_tokenizer, french_tokenizer = \
util.preprocess(english_sentences, french_sentences)

max_english_sequence_length = preproc_english_sentences.shape[1]
max_french_sequence_length = preproc_french_sentences.shape[1]
english_vocab_size = len(english_tokenizer.word_index)
french_vocab_size = len(french_tokenizer.word_index)

print('Data Preprocessed')
print("Max English sentence length:", max_english_sequence_length)
print("Max French sentence length:", max_french_sequence_length)
print("English vocabulary size:", english_vocab_size)
print("French vocabulary size:", french_vocab_size)

"""
### RNN Model
"""

# Reshaping the input to work with a basic RNN
tmp_x = util.pad(preproc_english_sentences, max_french_sequence_length)
tmp_x = tmp_x.reshape((-1, preproc_french_sentences.shape[-2], 1))

#create logger
simple_model_logger = CSVLogger("simple_model.log")

# Train the neural network
simple_rnn_model = models.rnn_model(
    tmp_x.shape,
    max_french_sequence_length,
    english_vocab_size,
    french_vocab_size)

simple_rnn_model.fit(tmp_x, preproc_french_sentences, batch_size=128, epochs=10, validation_split=0.2, 
                     callbacks=[simple_model_logger])

print(util.logits_to_text(simple_rnn_model.predict(tmp_x[:1])[0], french_tokenizer))

"""
### Embedding Model
"""

tmp_x = util.pad(preproc_english_sentences, max_french_sequence_length)
tmp_x = tmp_x.reshape((-1, preproc_french_sentences.shape[-2]))

embed_model_logger = CSVLogger("embed_model.log")

embedding_model = models.embedding_rnn_model(tmp_x.shape, max_french_sequence_length, english_vocab_size, 
                              french_vocab_size)

embedding_model.fit(tmp_x, preproc_french_sentences, batch_size=128, epochs=10, 
                    validation_split=0.2, callbacks=[embed_model_logger])

print(util.logits_to_text(embedding_model.predict(tmp_x[:1])[0], french_tokenizer))

"""
### Bidirectional RNN
"""

tmp_x = util.pad(preproc_english_sentences, max_french_sequence_length)
tmp_x = tmp_x.reshape((-1, preproc_french_sentences.shape[-2]))

tmp_x = util.pad(preproc_english_sentences, max_french_sequence_length)
tmp_x = tmp_x.reshape((-1, preproc_french_sentences.shape[-2], 1))

bd_model_logger = CSVLogger("bd_model.log")

bd_rnn_model = models.bidirectional_rnn_model(tmp_x.shape, max_french_sequence_length, english_vocab_size, french_vocab_size)
bd_rnn_model.fit(tmp_x, preproc_french_sentences, batch_size=128, epochs=10, 
                 validation_split=0.2, callbacks=[bd_model_logger])

print(util.logits_to_text(bd_rnn_model.predict(tmp_x[:1])[0], french_tokenizer))

"""
### Bidirectional RNN with Embeddings
"""

tmp_x = util.pad(preproc_english_sentences, max_french_sequence_length)
tmp_x = tmp_x.reshape((-1, preproc_french_sentences.shape[-2]))

tmp_x = util.pad(preproc_english_sentences, max_french_sequence_length)
tmp_x = tmp_x.reshape((-1, preproc_french_sentences.shape[-2]))

bd_model_with_embed_logger = CSVLogger("bd_model_with_embed.log")

bd_rnn_model = bidirectional_rnn_with_emb_model(tmp_x.shape, max_french_sequence_length, 
english_vocab_size, french_vocab_size)

bd_rnn_model.fit(tmp_x, preproc_french_sentences, batch_size=128, epochs=10, 
                 validation_split=0.2, callbacks=[bd_model_with_embed_logger])

print(util.logits_to_text(bd_rnn_model.predict(tmp_x[:1])[0], french_tokenizer))

"""
### Encoder Decoder Model
"""

# OPTIONAL: Train and Print prediction(s)
tmp_x = util.pad(preproc_english_sentences, max_english_sequence_length)
tmp_x = tmp_x.reshape((-1, preproc_english_sentences.shape[1], 1))

encdec_model_logger = CSVLogger("encdec_model.log")

encdec_rnn_model = models.encoder_decoder_model(input_shape, output_sequence_length, english_vocab_size, french_vocab_size)(tmp_x.shape, max_english_sequence_length, english_vocab_size, french_vocab_size)
encdec_rnn_model.fit(tmp_x, preproc_french_sentences, batch_size=128, epochs=10, validation_split=0.2, callbacks=[encdec_model_logger])

print(util.logits_to_text(encdec_rnn_model.predict(tmp_x[:1])[0], french_tokenizer))

"""
### Encoder Decoder with Embeddings Model
"""

tmp_x = util.pad(preproc_english_sentences, max_french_sequence_length)
tmp_x = tmp_x.reshape((-1, preproc_french_sentences.shape[-2]))

tmp_x = util.pad(preproc_english_sentences, max_french_sequence_length)
tmp_x = tmp_x.reshape((-1, preproc_french_sentences.shape[-2]))

model_final_logger = CSVLogger("model_final.log")

last_model = models.encoder_decoder_embeddings_model(input_shape, output_sequence_length, english_vocab_size, french_vocab_size)model_final(tmp_x.shape, max_french_sequence_length, english_vocab_size, french_vocab_size)
last_model.fit(tmp_x, preproc_french_sentences, batch_size=128, epochs=10, 
               validation_split=0.2, callbacks=[model_final_logger])

print(util.logits_to_text(last_model.predict(tmp_x[:1])[0], french_tokenizer))

"""
### Final Predictions (Selected model to train and infer)
"""

def final_predictions(x, y, x_tk, y_tk):
    """
    Gets predictions using the final model
    :param x: Preprocessed English data
    :param y: Preprocessed French data
    :param x_tk: English tokenizer
    :param y_tk: French tokenizer
    """
    model = model_final(x.shape, max_french_sequence_length, english_vocab_size, french_vocab_size)
    model.fit(x, y, batch_size=128, epochs=10, validation_split=0.2)
    
    y_id_to_word = {value: key for key, value in y_tk.word_index.items()}
    y_id_to_word[0] = '<PAD>'

    sentence = 'he saw a old yellow truck'
    sentence = [x_tk.word_index[word] for word in sentence.split()]
    sentence = pad_sequences([sentence], maxlen=x.shape[-1], padding='post')
    sentences = np.array([sentence[0], x[0]])
    predictions = model.predict(sentences, len(sentences))

    print('Sample 1:')
    print(' '.join([y_id_to_word[np.argmax(x)] for x in predictions[0]]))
    print('Il a vu un vieux camion jaune')
    print('Sample 2:')
    print(' '.join([y_id_to_word[np.argmax(x)] for x in predictions[1]]))
    print(' '.join([y_id_to_word[np.max(x)] for x in y[0]]))


final_predictions(preproc_english_sentences, preproc_french_sentences, 
                  english_tokenizer, french_tokenizer)
