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

from os import path



# Function for generating parsing rule list.
###################################################################################################################################################################################

def NP_list():
    parser_data = pd.read_csv('dataset/parser.csv')
    NP = np.array(parser_data) 
    NP = np.unique(NP)
    NP = list(NP)

    return (NP)





# Function for generating continuous.
###################################################################################################################################################################################

def get_continuous_chunks(text, chunk_func = ne_chunk):
    
    chunked = chunk_func(pos_tag(word_tokenize(text)))
    
    continuous_chunk = []
    current_chunk = []

    for subtree in chunked:
        
        if type(subtree) == Tree:
            current_chunk.append(" ".join([token for token, pos in subtree.leaves()]))
            
            continuous_chunk.append(current_chunk)
            current_chunk = []
        else:
            continue

    return continuous_chunk





# generating word list with parser rule.
###################################################################################################################################################################################

def n_gram_parser(df):
    
    resultList_uni = []
    chunker = []
    
    NP = NP_list()
        
    for i in range (0, len(NP)):
        chunker = chunker + [RegexpParser(NP[i])]
    
    get_data = []
    
    for i in range (0, len(chunker)):
        get_data = get_data + [df['text'].apply(lambda sent: get_continuous_chunks(sent, chunker[i].parse))]
        
        resultList_uni = resultList_uni + get_data[i][0]

    return (resultList_uni)




# converting pdf to text.
###################################################################################################################################################################################

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




# Removal of punctuation and stop word.
###################################################################################################################################################################################

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




# Prediction.
###################################################################################################################################################################################

def prediction(text_data, model):

    # converting to lowercase.
    #------------------------------------------------------------------------------
    text_data = text_data.lower()
    
    
    # removing stop word and punctuation.
    #------------------------------------------------------------------------------
    #text_data = preprocess_text(text_data) 
    
    
    # feedind data to n_gram_parser.
    #------------------------------------------------------------------------------
    df = pd.DataFrame({'text':[text_data]})
    word_list = n_gram_parser(df)
    
    
    # flattening into 1D.
    #------------------------------------------------------------------------------
    word_list = reduce(lambda x,y : x+y, word_list)
    
    
    # removing duplicates.
    #------------------------------------------------------------------------------
    word_list = list(dict.fromkeys(word_list))    
    
    
    tech_word = []
    
    for word in range(0, len(word_list)):
        
        try:
            pre_word = model.wv.most_similar(positive=[word_list[word]])
            tech_word = tech_word + [word_list[word]]
        except:
            pass
            
    if(len(tech_word)==0):
        return('no technical word found.')
    else:
        return(tech_word)
        
        
        
        
###################################################################################################################################################################################

# loading my saved model.
#------------------------------------------------------------------------------
model = Word2Vec.load('model/ankit_model.bin')
        

## Loading pdf file for prediction
##------------------------------------------------------------------------------
file_loc = 'c.pdf' 

try:
    text_data = text_from_pdf(file_loc)
    prediction_list = prediction(text_data, model)
    print('[TECHNICAL WORD]\n\n', prediction_list)

except:
    print('file cannot be converted.')

    
## Loading hardcoded string for prediction
##------------------------------------------------------------------------------
#text_data = "I @ his artificial neural networks artificial intelligence, reinforcement learning used c++ machine learning artificial artificial intelligence artificial "+\
#           "intelligence ajax javascript over QT Framework in order to get data data analytics and big data a nice GUI for what will be android further be an Django Project "+\
#           "with Python etc. Bachelor's degree in Finance Excellent experience of handling Business Intelligence Tools and Dashboard Reports Skilled at "+\
#           "consolidating and analyzing Financial Data Highly capable of Budgeting 3000 dollar, "
#prediction_list = prediction(text_data, model)
#print('[TECHNICAL WORD]\n\n', prediction_list)


# Loading excel for prediction
#------------------------------------------------------------------------------

    
# list containing prediction value.
#------------------------------------
#data_set = pd.read_csv('dataset/tech_summary.csv')
#
#for i in range(0, len(data_set)):
#    
#    text_data = data_set.summary[i]
#    print(text_data)
#    print('\n')
#    prediction_list = prediction(text_data, model)
#    print('[TECHNICAL WORD]\n\n', prediction_list)
#    print('\n\n')















