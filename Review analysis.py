

import pandas as pd

data = pd.read_csv('D:\datasets\IMDB Dataset.csv').dropna()
#data = data.head(2000)

d1 = data.review
d2 = data.sentiment

# for integer scoring
# def senti(n):
#     if n <3:
#         return 'negative'
#     elif n ==3:
#         return 'neutral'
#     else:
#         return 'positive'
# sentiment = []
# for i in d2:
#     sentiment.append(senti(i))

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(d1,d2,test_size = 0.2,random_state = 7)


from sklearn.feature_extraction.text import CountVectorizer 
vector = CountVectorizer()
trainv = vector.fit_transform(xtrain)
testv = vector.transform(xtest)

# print(xtrain[0])
# print(vv[0])

from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(random_state=1)
clf.fit(trainv,ytrain)
aa = clf.predict(testv)

ytest = ytest.values
cnt = 0
for i in range(0,len(aa)):
    if aa[i] == ytest[i]:
        cnt+=1
print('accuracy',cnt/len(ytest)*100)


ww = input("enter your review ")
test_set = [ww]
new_test = vector.transform(test_set)

print(clf.predict(new_test))

