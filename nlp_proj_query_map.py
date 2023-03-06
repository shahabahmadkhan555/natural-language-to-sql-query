#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import re
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import OrderedDict
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import string
import numpy
import pickle


def data_extraction(file):
	data=[]
	sql_query=[]
	corresponding_value=[]
	question_query={}

	texti=open(file,"r")
	text=texti.read()
	text=text.split("\n")
	text=text[0:len(text)-1]
	#print(text)
	dic={}
	for i in text:
		val=i.split(",")
		#print val
		data.append(val[0])
		if val[2]=='1':
			question_query[val[0]]=val[1]

	li=set()
	li=set(data)
	data=[]
	
	s=set(string.punctuation)
	data=list(li)
	train_data=[]
	num=0
	for value in data:
		dic[str(num)]=value
		num+=1
		tokens = word_tokenize(value)
		stop_words=(set(stopwords.words('english')))
		word = [t for t in tokens if not t in stop_words and not t in s]
		train_data.append(word)
	
	dbfile = open("question_dict", 'wb') 
	pickle.dump(dic, dbfile)
	dbfile.close()
	dbfile = open("question_query", 'wb') 
	pickle.dump(question_query, dbfile)
	dbfile.close()

	return train_data
	

	

def data_training(l):
	taged_data=[]
	for i in range(0,len(l)):
		taged_data.append(TaggedDocument(l[i],[str(i)]))
	#print taged_data
	model = Doc2Vec(vector_size=300,alpha=.041,min_alpha=0.041,min_count=0,dm =1,window=5)
	model.build_vocab(taged_data)
	for epoch in range(200):
		#print('iteration {0}'.format(epoch))
		model.train(taged_data,total_examples=len(taged_data),epochs=model.iter)
		model.alpha -= 0.0002
		model.min_alpha = model.alpha

	model.save("d2v.model")

def data_extraction2(file):
	data=[]
	texti=open(file,"r")
	text=texti.read()
	text=text.split("\n")
	text=text[0:len(text)-1]
	#print text
	dic={}
	for i in text:
		val=i.split(",")
		data.append(val[0])
		

	li=set()
	li=set(data)
	data=[]
	
	s=set(string.punctuation)
	data=list(li)
	train_data=[]
	num=0
	for value in data:
		dic[str(num)]=value
		num+=1
		tokens = word_tokenize(value)
		stop_words=(set(stopwords.words('english')))
		word = [t for t in tokens if not t in stop_words and not t in s]
		train_data.append(word)
	return train_data,dic




def test_data():
    stmt=[]
    f=open('question_dict',"rb")
    d=pickle.load(f)
    f=open('question_query',"rb")
    ques_query=pickle.load(f)

    label=[]
    f_stmt=[]
    test_list,sent=data_extraction2("D:\\Mtech@iiit\\I sem\\NLP\\NLP_PROJECT\\d2v_test")
	#print sent
    model=Doc2Vec.load("d2v.model")
    for i in range(0,len(test_list)):
        a1=model.infer_vector(test_list[i])
        a2=model.docvecs.most_similar([a1],topn = 5)   #contains all 5 top matching questions from train set
		#print a2
        stmt=[]
        #print("Question :"+sent[str(i)])
        for val in a2:
            
            st=d[str(val[0])]+","+ques_query[d[str(val[0])]]
            stmt.append(st)
          
        f_stmt.append(stmt)
   # print(f_stmt)
    #print(label)
    return f_stmt,label


	#print test_list


# In[2]:


def give_query(train_X,label):
    y=[]
    from tensorflow.keras.models import load_model
    model=load_model("sql_model.h5")
   # model.summary()
    y_pred=model.predict(train_X)
    y=y_pred.tolist()
    #print(y)
    pos=y.index(max(y))
    #print(pos)
    return(pos)
   


# In[3]:


l=data_extraction("D:\\Mtech@iiit\\I sem\\NLP\\NLP_PROJECT\\d2v_train")
data_training(l)
import numpy as np
f_stmt,label=test_data()
#print(f_stmt)
for stmt in f_stmt:
   # print(stmt)
    X_train=np.array(stmt)
    label=np.array(label)
    from tensorflow.keras.preprocessing.text import Tokenizer
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(X_train)
    train_X = tokenizer.texts_to_sequences(X_train)
    vocab= len(tokenizer.word_index) + 1
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    maxlen = 100
    train_X = pad_sequences(train_X, padding='post', maxlen=maxlen)

    p=give_query(train_X,label)
    X=X_train.tolist()

    s=X[p].split(',')
    print("\n")
    print("Question:",s[0])
    print("Mapped Query:",s[1])

    


# In[ ]:





# In[ ]:




