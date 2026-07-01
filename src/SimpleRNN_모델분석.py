from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN
import numpy as np

x = np.array( [[5,6,7],[1,2,3],[11,12,13],[6,7,8]] )
print(x)
y = np.array( [8,4,14,9] )

rnnmodel = Sequential()
rnnmodel.add( SimpleRNN(10,return_sequences=False,input_shape=(3,1)) )
rnnmodel.add( Dense(1) )
rnnmodel.summary()

# 모델 컴파일
rnnmodel.compile(loss = 'mse' , optimizer = 'adam' , metrics =
                 ['mse'] )

rnnmodel.fit(x, y , epochs = 1000, batch_size = 1 )