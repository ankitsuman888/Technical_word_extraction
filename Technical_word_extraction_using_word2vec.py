#!/usr/bin/env python
# coding: utf-8

# # Importing Packages.

# In[1]:


from gensim.models import Word2Vec
import pandas as pd

import numpy as np
from functools import reduce

import nltk

from nltk import word_tokenize, pos_tag, ne_chunk
from nltk import RegexpParser
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk import Tree

import PyPDF2 
import textract
import re

from os import path

from sklearn.model_selection import train_test_split

import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation, SpatialDropout1D
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
from keras.layers.embeddings import Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from functools import reduce
from keras.utils import to_categorical
from keras.models import load_model   
from keras.preprocessing.text import one_hot


# # Loading data from word2vec Corpus.

# #### Loading corpus and creating vectorized data. 

# In[2]:



def word_vectors_list():
    
    vector_model = Word2Vec.load('ankit_word2vec_model.bin')
    dataset = list(vector_model.wv.vocab)

    data_vector_list = []

    for i in range(0, len(dataset)):
        try:
            data  = [[str(dataset[i])]]

            model = Word2Vec(data, min_count = 1, size=100)
            data_word  = model[dataset[i]] 

            data_vector_list = data_vector_list + [data_word]

        except:
            pass
        
    return(data_vector_list)


# In[3]:


def word_list():
    
    vector_model = Word2Vec.load('ankit_word2vec_model.bin')
    data_list = list(vector_model.wv.vocab)
        
    return(data_list)


# # N-gram generation.
# #### Function for generating n-gram.

# In[4]:



def generate_ngrams(s, n):
    
    # Convert to lowercases.
    #--------------------------------------------------------
    s = s.lower()
    
    # Replace all none alphanumeric characters with spaces.
    #--------------------------------------------------------
    s = re.sub(r'[^a-zA-Z0-9\s+-]', '', s)
    
    # Break sentence in the token, remove empty tokens.
    #--------------------------------------------------------
    token = [token for token in s.split(" ") if token != ""]
    
    # Use the zip function to help us generate n-grams
    # Concatentate the tokens into ngrams and return.
    #--------------------------------------------------------
    ngrams = zip(*[token[i:] for i in range(n)])
    
    return [" ".join(ngram) for ngram in ngrams]


# # Data Preprocessing.
# 

# #### Removal of stop word and punctuations.

# In[5]:



def preprocess_text(sentence):
    sentence = sentence.lower()
    
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(sentence)
    
    # custom stopword list.
    #----------------------------
    stop_word_data = pd.read_csv('stopwords/sw3.txt')
    stop_word_data = list(stop_word_data['stopwords'])
    
    # filtered_words = [w for w in tokens if not w in stopwords.words('english')]
    filtered_words = [w for w in tokens if not w in stop_word_data]
    
    return " ".join(filtered_words)


# # Generating processed data.
# #### removing stop word and creating n_grams.

# In[6]:



def data_formation(data):
    
    text_data = []
    
    data = preprocess_text(data)

    uni_gram_data = generate_ngrams(data, 1)
    bi_gram_data  = generate_ngrams(data, 2)
    tri_gram_data = generate_ngrams(data, 3)
    
    text_data = uni_gram_data + bi_gram_data + tri_gram_data
    
    text_data = list(dict.fromkeys(text_data))
    
    return(text_data)


# # Converting pdf to text.

# In[7]:



def text_from_pdf(file_loc):
   
    filename = file_loc

    # open allows you to read the file.
    #------------------------------------
    pdfFileObj = open(filename,'rb')

    # The pdfReader variable is a readable object that will be parsed.
    #-----------------------------------------------------------------
    pdfReader = PyPDF2.PdfFileReader(pdfFileObj)

    # Discerning the number of pages will allow us to parse through all #the pages.
    #-----------------------------------------------------------------------------
    num_pages = pdfReader.numPages
    count = 0
    text = ""

    # The while loop will read each page.
    #------------------------------------
    while count < num_pages:
        pageObj = pdfReader.getPage(count)
        count +=1
        text += pageObj.extractText()

    # This if statement exists to check if the above library returned #words. It's done because PyPDF2 cannot read scanned files.
    #----------------------------------------------------------------------------------------------------------------------------
    if text != "":
        text = text

    #If the above returns as False, we run the OCR library textract to #convert scanned/image based PDF files into text
    #------------------------------------------------------------------------------------------------------------------
    else:
        text = textract.process(filename, method='tesseract', language='eng')
 
    return(text)


# # Data Encoding.
# #### Finding emmbed_dimension, pad_sequence, vocab_size.

# In[8]:



def encoded_data(model_data) :
        
    # emmbedding dimension.
    #-----------------------
    EMBEDDING_DIM = 100
    print('embedding dimension: ', EMBEDDING_DIM)
    
    # pad sequence { total number of words in a single phrase }.
    #----------------------------------------------------------------
    max_length = max([len(s.split()) for s in model_data])
    print('pad sequence :', max_length)

    # define vocabulary size
    #----------------------------------------------------------------
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(model_data)
    vocab_size = len(tokenizer.word_index) + 1
    print('vocabulary size :', vocab_size)

    # encoding and pad documents to a max length of 4 words'.
    #---------------------------------------------------------
    encoded_docs = [one_hot(d, vocab_size) for d in model_data]
    padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
    
    return([EMBEDDING_DIM, max_length, vocab_size, padded_docs])


# In[9]:



def vectorization(word_list, sav_non_tech):    
    
    vec_word = []
    model = Word2Vec.load('ankit_word2vec_model.bin')
    word_list = data_formation(word_list) 
        
    for word in range(0, len(word_list)):
        
        try:
            vec_tech_word = model.wv.most_similar(positive=[word_list[word]])
            vec_word = vec_word + [word_list[word]]
        except:
            if(sav_non_tech == True):
                saving_non_tech_word(word_list[word])
            else:
                pass
            
    if(len(vec_word)==0):
        return('no technical word found.')
    else:        
        return(vec_word)
    


# # Saving parameters.
# #### Saving max-length and vocab size. 

# In[10]:



def saving_parameters(max_length, vocab_size):
    
    if(path.exists('dataset/param_data.csv') == True):
        with open('dataset/param_data.csv', 'a') as csvFile:
            csvFile.write('\n{},{}'.format(max_length, vocab_size))        
        
    else:
        with open('dataset/param_data.csv', 'w') as csvFile:
            csvFile.write('{},{}'.format('max_length','vocab_size'))
            csvFile.write('\n{},{}'.format(max_length, vocab_size))
    


# # Model Training.

# #### Deep Learninig model.

# In[11]:



def Model( vocab_size, EMBEDDING_DIM, max_length):
   
    # Now defining our model.
    #-------------------------
    model = Sequential()
    model.add(Embedding(vocab_size, EMBEDDING_DIM, input_length=max_length, trainable=True))
    model.add(SpatialDropout1D(0.2))
    model.add(LSTM(256))
    model.add(Dense(128))
    model.add(Dense(64))
    model.add(Dense(32))
    model.add(Dense(1, activation='sigmoid'))

    # compiling our model.
    #-----------------------
    adam_opt = keras.optimizers.Adam(lr=0.0001)
    model.compile(loss='binary_crossentropy', optimizer = adam_opt, metrics=['accuracy'])
    
    return model


# In[12]:



def training():
    
    # loading data from corpus.
    #------------------------------------------------------------------
    model_data_val = word_list()
    model_data_labels = [1]*len(model_data_val)
    model_data_1 = pd.DataFrame(list(zip(model_data_val, model_data_labels)), columns =['Phrases', 'Labels'])
    model_data_2 = pd.read_csv('dataset/non_tech_data/train_non_tech_data.csv')

    model_data = model_data_1.append(model_data_2, ignore_index=True)
    
    
    # model history.
    #-----------------
    model_history = []
    
    # getting encoded data info.
    #----------------------------------
    encode_data = encoded_data(model_data.Phrases)
    
    EMBEDDING_DIM = encode_data[0]    
    max_length = encode_data[1]
    vocab_size = encode_data[2]
    padded_docs = encode_data[3]
    
    model = Model(vocab_size, EMBEDDING_DIM, max_length)
    model.summary()
    
    # fitting data to model.
    #-----------------------------------------------------------------------
    model.fit(padded_docs, model_data.Labels, epochs=10, batch_size = 256, verbose=2) 
    
    # saving trained model.
    #------------------------------------------------------------------------
    model.save('tech_model.h5')
    
    print("\nModel trained and saved successfully.")
       
    # saving max-length and vocab-size for prediction.
    #------------------------------------------------
    saving_parameters(max_length, vocab_size)


# # Training part.

# In[13]:


training()


# # Creating CSV for non-technical word.
# #### Saving the non technical word into csv with label 0.  So, that we can use this csv to train the model.

# In[14]:



def saving_non_tech_word(word):
        
    if(path.exists('dataset/non_tech_data/train_non_tech_data.csv') == True):
        with open('dataset/non_tech_data/train_non_tech_data.csv', 'a') as csvFile:
            csvFile.write('\n{},{}'.format(word, 0))        
        
    else:
        with open('dataset/non_tech_data/train_non_tech_data.csv', 'w') as csvFile:
            csvFile.write('{},{}'.format('Phrases','Labels'))
            csvFile.write('\n{},{}'.format(word , 0))
                               


# # Function calling.
# ### Prediction result.

# ### Generating output for technical and non technical word.

# In[15]:



def prediction(model_name, text_data, decision_boundary, sav_non_tech):
    
    # loading the saved model.
    #---------------------------
    model = load_model(model_name)
    
    # loading model parameters from csv.
    #----------------------------------
    try:        
        param_data = pd.read_csv('dataset/param_data.csv')
        param_data = param_data.tail(1)
        max_length = int(param_data.max_length)
        vocab_size = int(param_data.vocab_size)
    except:
        print('Error reading param file. please check.')
        return
      
    try:
        
        finallist = data_formation(text_data)

        # defining list.
        #----------------
        final_pred_tech = []

        # encoding.
        #-----------
        encoded_docs = [one_hot(d, vocab_size) for d in finallist]
        
        # data processing and formation.
        #---------------------------------------------------------------------
        vectorization_list = vectorization(text_data, sav_non_tech)
                
        # pad documents to a max length of 4 words'
        # -----------------------------------------
        padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
  
        for i in range(0, len(padded_docs)):
            data = padded_docs[i]    
           
            get_data = model.predict([[data]])

            # decision boundary is increase and decrease as required manually.
            #--------------------------------------------------------------------------------
            if (get_data[0][0] > decision_boundary):
                final_pred_tech.append(finallist[i])               

        final_pred_tech.extend(vectorization_list)
        final_pred_tech = list(dict.fromkeys(final_pred_tech))
        
        return(final_pred_tech)
    
    except Exception as e:
        print('Error : may be empty string.\n\n',e)


# ### Prediction For pdf file.

# In[22]:



# select the pdf location.
#-----------------------------------------
try:
    file_loc = 'testing_data/a.pdf' 
    text_data = text_from_pdf(file_loc)
except:
    print('error occured during conversion.')

# select the model here after traininig.
#-----------------------------------------
model_name = 'tech_model.h5'

# decision boundary {currently 80%}.
# please adjust the decision boundary as required.
#-----------------------------------------------
decision_boundary = 0.90    
    
# saving non_tech word.
#------------------------------------
sav_non_tech = False
    
# list containing prediction value.
#------------------------------------

prediction_list = prediction(model_name, text_data, decision_boundary, sav_non_tech)
print('[TECHNICAL WORD]\n\n', prediction_list)


# ### Prediction For Single String. 

# In[20]:


# string.
#-----------------------------------------
text_data = "I @ this artificial neural networks artificial intelligence, reinforcement learning used c++ machine learning artificial artificial intelligence artificial "+           "intelligence ajax javascript over QT Framework in order to get data data analytics and big data a nice GUI for what will be android further be an Django Project "+           "with Python etc. Bachelor's degree in Finance Excellent experience of handling Business Intelligence Tools and Dashboard Reports Skilled at "+           "consolidating and analyzing Financial Data Highly capable of Budgeting 3000 dollar, "

# select the model here after traininig.
#-----------------------------------------
model_name = 'tech_model.h5'

# decision boundary {currently 80%}.
# please adjust the decision boundary as required.
#-----------------------------------------------
decision_boundary = 0.90   

# saving non_tech word.
#------------------------------------
sav_non_tech = False

# list containing prediction value.
#------------------------------------

prediction_list = prediction(model_name, text_data, decision_boundary, sav_non_tech)
print('[TECHNICAL WORD]\n\n', prediction_list)


# In[ ]:





# In[ ]:




