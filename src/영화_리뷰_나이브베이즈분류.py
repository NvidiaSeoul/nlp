import numpy as np
import pandas as pd
# pd.set_option('display.max_rows',1000)
# pd.set_option('display.max_columns', 500)
# pd.set_option('display.width',1000)
# pd.set_option('max_colwidth', 1000)

np.set_printoptions(threshold=np.inf) #무한으로 출력합니다.

Moviedf = pd.read_csv('/home/sckit/deeplearning_prj/20260616/IMDB Dataset.csv')
#Moviedf.info()

import re
def MovieReviewControl(arg):
    return re.sub(r'[^a-zA-z\s]','',arg)

Moviedf['review'] = Moviedf['review'].apply(MovieReviewControl)


# sentiment  컬럼 라벨을 수치 데이터로 변경
Moviedf['sentiment'] = Moviedf['sentiment'].map({'positive':1,'negative':0})
#print(Moviedf)

train_x = Moviedf['review']
train_y = Moviedf['sentiment']

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
train_cv = cv.fit_transform(train_x)
print(train_cv)
#train_cv_encoded = train_cv.toarray()
#print(train_cv_encoded)
#print(train_cv_encoded[0])
print(len(cv.get_feature_names_out()))
# print(cv.get_feature_names_out())

from sklearn.naive_bayes import MultinomialNB # 다항분포 나이브베이즈
mnb = MultinomialNB()
mnb.fit(train_cv, train_y)

print('acc : ', mnb.score(train_cv, train_y))


# 새로운 영화 리뷰 데이터 입력해서 예측 ?
