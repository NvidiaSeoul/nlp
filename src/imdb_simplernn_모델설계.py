
from tensorflow.keras.datasets import imdb
import numpy as np
(train_x, train_y),( test_x , test_y ) = imdb.load_data(num_words = 500) # 내부적으로 0,1,2,3 은 특수토근
print(len(train_x)) # 25000
print(len(test_x)) # 25000
print(train_x[0])
# 타깃이진분류 : 0(부정) / 12500개, 1(긍정) / 12500
print(np.unique(train_y,return_counts=True )) # 라벨(정답)도 수치형으로 변환되어있음

from sklearn.model_selection import train_test_split

train_x, val_x, train_y, val_y = \
    train_test_split(train_x, train_y, test_size=0.2, random_state=43)

print(len(train_x)) # 20000
print(len(val_x)) # 5000

print(len(train_x[0]))
print(len(train_x[1]))

# 1차 ==> 길이가 다른 리뷰 정수데이터 배열을  길이가 동일한 정수 배열로 변경
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 길이를 100로 모두 변경할때  짧은거는 0으로 채우고, 긴거는 버린다!
train_seq = pad_sequences(train_x, maxlen = 100)
print(train_seq[0])
print(train_seq.shape)

val_seq = pad_sequences(val_x, maxlen = 100)
print(val_seq[0])
print(val_seq.shape)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN,Dense, Input

rnnmodel = Sequential()
input = Input( shape=(100,) )
rnnmodel.add(input)
rnnmodel.add( Embedding(input_dim=500, output_dim=16, input_length=100 ))
rnnmodel.add( SimpleRNN(8) )
rnnmodel.add( Dense(1 , activation='sigmoid'))
rnnmodel.summary()

# train_x ==> 정수벡터화 되어 있음

# word_index = imdb.get_word_index() # 어떤 단어를 어떤 수치로 변환했는지에 대한 정보를 가지고 있음
# #print(word_index)

# for word, index in word_index.items():
#     #print(word, index)  # 빈도수가 가장 높은거는 index ==> 1
#     if index == 1:
#         print(word)

# # train_x[0] ==> 정수벡터를  ==> 단어들의 모음으로 치환시키는 작업
# conv_word_index =dict( [ (idx+3, word) for (word, idx) in word_index.items() ] )
# print(conv_word_index)

# # for word, idx in conv_word_index.items():
# #     if idx == 4:
# #         print(word)

# decode_sentence = ' '.join( [  conv_word_index[i]  if i in conv_word_index else '?' for i in train_x[0]  ] )
# print(decode_sentence)