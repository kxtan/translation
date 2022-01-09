from keras.models import Model
from keras.layers import GRU, Input, Dense, TimeDistributed, Activation, RepeatVector, Bidirectional
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from keras.losses import sparse_categorical_crossentropy


def rnn_model(input_shape, output_sequence_length, 
    english_vocab_size, french_vocab_size, lr=1e-3):
    """
    Build and train a basic RNN on x and y
    :param input_shape: Tuple of input shape
    :param output_sequence_length: Length of output sequence
    :param english_vocab_size: Number of unique English words in the dataset
    :param french_vocab_size: Number of unique French words in the dataset
    :return: Keras model built, but not trained
    """
  
    inputs = Input(shape=input_shape[1:])
   
    x = GRU(128, return_sequences=True)(inputs)
    x = TimeDistributed(Dense(french_vocab_size, activation="softmax"))(x)
    
    model = Model(inputs, Activation('tanh')(x))
    
    model.compile(loss=sparse_categorical_crossentropy,
                  optimizer=Adam(lr),
                  metrics=['accuracy'])
    
    return model


def embedding_rnn_model(input_shape, output_sequence_length, 
    english_vocab_size, french_vocab_size, lr=1e-3):
    """
    Build and train a RNN model using word embedding on x and y
    :param input_shape: Tuple of input shape
    :param output_sequence_length: Length of output sequence
    :param english_vocab_size: Number of unique English words in the dataset
    :param french_vocab_size: Number of unique French words in the dataset
    :return: Keras model built, but not trained
    """
    
    inputs = Input(shape=(input_shape[1:]))
    
    x = Embedding(input_dim=english_vocab_size, output_dim=64)(inputs)
    x = GRU(128, return_sequences=True)(x)  
    x = TimeDistributed(Dense(french_vocab_size, activation='softmax'))(x)

    model = Model(inputs, Activation('tanh')(x))
    model.compile(loss=sparse_categorical_crossentropy,
                  optimizer=Adam(lr),
                  metrics=['accuracy'])
    
    return model


def bidirectional_rnn_model(input_shape, output_sequence_length, 
    english_vocab_size, french_vocab_size, lr=1e-3):
    """
    Build and train a bidirectional RNN model on x and y
    :param input_shape: Tuple of input shape
    :param output_sequence_length: Length of output sequence
    :param english_vocab_size: Number of unique English words in the dataset
    :param french_vocab_size: Number of unique French words in the dataset
    :return: Keras model built, but not trained
    """
    
    inputs = Input(input_shape[1:])
    
    x = Bidirectional(GRU(128, return_sequences=True))(inputs)
    x = TimeDistributed(Dense(french_vocab_size, activation='softmax'))(x)
    
    model = Model(inputs, Activation('tanh')(x))
    model.compile(loss=sparse_categorical_crossentropy,
                  optimizer=Adam(lr),
                  metrics=['accuracy'])
    
    return model


def bidirectional_rnn_with_emb_model(input_shape, output_sequence_length, 
    english_vocab_size, french_vocab_size, lr=1e-3):
    """
    Build and train a bidirectional RNN model on x and y
    :param input_shape: Tuple of input shape
    :param output_sequence_length: Length of output sequence
    :param english_vocab_size: Number of unique English words in the dataset
    :param french_vocab_size: Number of unique French words in the dataset
    :return: Keras model built, but not trained
    """
 
    inputs = Input(input_shape[1:])
 
    x = Embedding(input_dim=english_vocab_size, output_dim=64)(inputs)
    x = Bidirectional(GRU(128, return_sequences=True))(x)
    x = TimeDistributed(Dense(french_vocab_size, activation='softmax'))(x)
    
    model = Model(inputs, Activation('tanh')(x))
    model.compile(loss=sparse_categorical_crossentropy,
                  optimizer=Adam(lr),
                  metrics=['accuracy'])
    
    return model


def encoder_decoder_model(input_shape, output_sequence_length, 
    english_vocab_size, french_vocab_size, lr=1e-3):
    """
    Build and train an encoder-decoder model on x and y
    :param input_shape: Tuple of input shape
    :param output_sequence_length: Length of output sequence
    :param english_vocab_size: Number of unique English words in the dataset
    :param french_vocab_size: Number of unique French words in the dataset
    :return: Keras model built, but not trained
    """
    
    inputs = Input(shape=(input_shape[1:]))
    
    encoder = GRU(128, return_sequences=True)(inputs)
    encoder = GRU(128)(encoder)
    encoder = RepeatVector(21)(encoder)

    decoder = GRU(128, return_sequences=True)(encoder)
    decoder = TimeDistributed(Dense(french_vocab_size, activation='softmax'))(decoder)
    
    model = Model(inputs = inputs, outputs = decoder)
    model.compile(loss=sparse_categorical_crossentropy,
                  optimizer=Adam(lr),
                  metrics=['accuracy'])
    
    return model


def encoder_decoder_embeddings_model(input_shape, output_sequence_length, 
    english_vocab_size, french_vocab_size, lr = 1e-3):
    """
    Build and train a model that incorporates embedding, encoder-decoder, and bidirectional RNN on x and y
    :param input_shape: Tuple of input shape
    :param output_sequence_length: Length of output sequence
    :param english_vocab_size: Number of unique English words in the dataset
    :param french_vocab_size: Number of unique French words in the dataset
    :return: Keras model built, but not trained
    """
    
    inputs = Input(shape=input_shape[1:])
    embedding_layer = Embedding(input_dim=english_vocab_size, output_dim=64)(inputs)
    
    encoder = Bidirectional(GRU(128, return_sequences=True))(embedding_layer)
    encoder = Bidirectional(GRU(128))(encoder)
    encoder = RepeatVector(output_sequence_length)(encoder)

    decoder = Bidirectional(GRU(128, return_sequences=True))(encoder)
    decoder = TimeDistributed(Dense(french_vocab_size, activation='softmax'))(decoder)
    
    model = Model(inputs=inputs, outputs=decoder)
    
    # Compile the model
    model.compile(loss=sparse_categorical_crossentropy,
                  optimizer=Adam(lr=learning_rate),
                  metrics=['accuracy'])
    
    return model