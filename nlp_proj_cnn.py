#!/usr/bin/env python
# coding: utf-8

# In[1]:


def read_file():						#function to read file		
    file="D:\\Mtech@iiit\\I sem\\NLP\\project\\train"
    import re
    f=open(file,'r',encoding='utf-8')
    stmt=[]
    lines=[]
    label=[]
    from nltk.corpus import stopwords 
    stop_words = set(stopwords.words('english')) 
    from gensim.parsing.preprocessing import remove_stopwords
    q=[]
    j=[]
    symb=['(',')','[',']','<','>','-','_','^','*','%','?',',',':',';','"'',''','.']
    no=[0,1]
    query=[]
    with open(file,'r') as f:
        for l in f:
            l.lower()
           
            lines.append(l)
   # print(lines)
    
    for i in lines:
        x=re.findall('[01]\n',i)
        i=remove_stopwords(i)
        j.append(x[0])
        i=i.strip(',0\n')
        i=i.strip(',1\n')
        stmt.append(i)
    
            
        f_stmt=[]
   
    for i in stmt:
        from nltk.tokenize import word_tokenize
        i= word_tokenize(i)  
        f_stmt.append(i)
            
    ques=[]
    
   # print(f_stmt)
   
            
    for k in j:
        label.append(float(k.strip('\n')))
    return f_stmt,label
   
    
    


# In[36]:


stmt,label=read_file()
import numpy as np

stmt=np.array(stmt)
label=np.array(label)
#print(len(label))
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(stmt,label,test_size=0.10)

from tensorflow.keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X_train)
train_X = tokenizer.texts_to_sequences(X_train)
test_X = tokenizer.texts_to_sequences(X_test)
vocab= len(tokenizer.word_index) + 1


from tensorflow.keras.preprocessing.sequence import pad_sequences
maxlen = 100
train_X = pad_sequences(train_X, padding='post', maxlen=maxlen)
test_X = pad_sequences(test_X, padding='post', maxlen=maxlen)
print(train_X.shape)
print("for neural network:-")
#nn(train_X,test_X,y_train,y_test,maxlen)
print("for cnn-")
cnn_query(train_X,test_X,y_train,y_test,maxlen)





