from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import validation_curve
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from numpy import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
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
transformer.fit(x)
x=transformer.transform(x)
y = np.array(y)
#print (transformer.get_feature_names())
#print (type(X.toarray()))
a=x.toarray()
rows = a.shape[0]
cols = a.shape[1]
#print (rows)
#print (cols)
ct=0
model = RandomForestClassifier(n_estimators=100,random_state=0,n_jobs=1)
random.seed(0)
score = cross_val_score(model, x, y, cv=3, scoring='roc_auc')

print ("Score",np.round(np.mean(score), 2))
#print ("STD", np.round(np.std(score), 3))

print()
print()

#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.75, random_state=0)
#model.fit(x_train,y_train)
#y_pred = model.predict(x_test)
#print ('accuracy',accuracy_score(y_test, y_pred, normalize=True))

param_range=np.arange(1,25)
train_scores, test_scores = validation_curve(model,x,y,param_name="n_estimators",param_range=param_range,cv=3,scoring='roc_auc')
#print train_scores
#print test_scores
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)
#print (train_scores_mean)
test_scores_mean[0]=0.5
print (test_scores_mean)
plt.title("Validation Curve with Random Forest")
plt.xlabel("number of decision trees")
plt.ylabel("Score")
plt.ylim(0.0, 1.1)
lw=2
plt.plot(param_range, train_scores_mean, label="Training score",color="orangered",lw=lw)
plt.fill_between(param_range, train_scores_mean - train_scores_std,train_scores_mean + train_scores_std,color="lightsalmon",lw=lw)
plt.plot(param_range, test_scores_mean, label="Cross-validation score",color="navy",lw=lw)
plt.fill_between(param_range, test_scores_mean - test_scores_std,test_scores_mean + test_scores_std,color="lightskyblue",lw=lw)
#plt.tight_layout()
plt.legend(loc="best")
plt.show()