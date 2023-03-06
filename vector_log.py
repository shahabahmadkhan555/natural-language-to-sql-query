from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
import numpy as np
from numpy import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import validation_curve
import matplotlib.pyplot as plt
fname="train"
x=[];y=[]
#w=open("queries",'w')
with open(fname) as f:
	for line in f:
		#print (line)
		record = line.split(',')

		x.append(record[0] + ' ' + record[1])
		y.append(int(record[-1]))
		#w.write(record[1]+"\n")
# create vectorizer
transformer = CountVectorizer(stop_words="english", binary=True)
#print (transformer)
x = transformer.fit_transform(x)
y = np.array(y)
print (transformer.get_feature_names())
#print (type(X.toarray()))
a=x.toarray()
rows = a.shape[0]
cols = a.shape[1]
#print (rows)
#print (cols)
ct=0
#for x in range(0, rows):
	#for y in range(0, cols):
		#print (a[x,y],end=" ")
	#print ()  
	#print ((a[x]==1).sum())
'''	
transformer = CountVectorizer(stop_words="english", binary=True)
#print (transformer)
X = transformer.fit_transform(X)
y = np.array(y)
print (transformer.get_feature_names())
print(X.toarray())
'''
'''
transformer = CountVectorizer(stop_words="english", binary=True)
#print (transformer)
X = transformer.fit_transform(X)
y = np.array(y)
print (transformer.get_feature_names())
print(X.toarray())
transformer = CountVectorizer(stop_words="english", binary=True)
#print (transformer)
X = transformer.fit_transform(X)
y = np.array(y)
print (transformer.get_feature_names())
print(X.toarray())
transformer = CountVectorizer(stop_words="english", binary=True)
#print (transformer)
X = transformer.fit_transform(X)
y = np.array(y)
print (transformer.get_feature_names())
print(X.toarray())
'''

clf=LogisticRegression(penalty='l1',solver='liblinear')
random.seed(0)
score = cross_val_score(clf, x, y, cv=3, scoring='roc_auc')

print ("Score",np.round(np.mean(score), 2))
print ("STD", np.round(np.std(score), 3))

print()
print()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
print ('accuracy',accuracy_score(y_test, y_pred, normalize=True))

'''
q=open("questins")
ques=q.read()
ques=ques.split('\n')
ques=ques[1:len(ques)-1]
a=pen("queries")
queries=a.read()
queries=queries.split('\n')
queries=queries[1:len(queries)-1]
x=[]
for question in ques:
	for query in queries:
		x.append(question+' '+queries)

'''

param_range = np.logspace(-4, 4, 20)
train_scores, test_scores = validation_curve(clf, x, y, param_name="C", param_range=param_range, cv=3, scoring="roc_auc", n_jobs=-1)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)
print (test_scores_mean)
plt.title("Validation Curve with Logistic Regression")
plt.xlabel("C")
plt.ylabel("Score")
plt.ylim(0.0, 1.1)
lw = 2
plt.semilogx(param_range, train_scores_mean, label="Training score",color="orangered",lw=lw)
plt.fill_between(param_range, train_scores_mean - train_scores_std,train_scores_mean + train_scores_std,color="lightsalmon",lw=lw)
plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",color="navy",lw=lw)
plt.fill_between(param_range, test_scores_mean - test_scores_std,test_scores_mean + test_scores_std,color="lightskyblue",lw=lw)
plt.legend(loc="best")
plt.show()