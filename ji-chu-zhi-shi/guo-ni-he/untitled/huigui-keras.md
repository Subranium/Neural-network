# 回归-keras

## 数据描述

波士顿房价数据集（Boston House Price Dataset）（[https://www.kaggle.com/c/boston-housing](https://www.kaggle.com/c/boston-housing)）

boston\_housing数据集对房价数据进行回归分析，数据来自1970年代，波斯顿周边地区的房价，是用于机器学习的经典数据集。该数据集很小，共计506条数据

每条数据包含房屋以及房屋周围的详细信息。其中包含城镇犯罪率，一氧化氮浓度，住宅平均房间数，到中心区域的加权距离以及自住房平均房价等等。

* CRIM：城镇人均犯罪率。
* ZN：住宅用地超过 25000 sq.ft. 的比例。
* INDUS：城镇非零售商用土地的比例。
* CHAS：查理斯河空变量（如果边界是河流，则为1；否则为0）。
* NOX：一氧化氮浓度。
* RM：住宅平均房间数。
* AGE：1940 年之前建成的自用房屋比例。
* DIS：到波士顿五个中心区域的加权距离。
* RAD：辐射性公路的接近指数。
* TAX：每 10000 美元的全值财产税率。
* PTRATIO：城镇师生比例。
* B：1000（Bk-0.63）^ 2，其中 Bk 指代城镇中黑人的比例。
* LSTAT：人口中地位低下者的比例。
* MEDV：自住房的平均房价，以千美元计。

## keras普通实现

```python
import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import boston_housing
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping

from sklearn.preprocessing import StandardScaler


def load_data():
    (x_train, y_train), (x_test, y_test) = boston_housing.load_data()
    x = np.vstack((x_train,x_test))
    y = np.concatenate((y_train, y_test))
    x = StandardScaler().fit_transform(x)
    y = StandardScaler().fit_transform(y.reshape(-1, 1))
    x_train = x[1:401, :]
    x_test = x[401:, :]
    y_train = y[1:401, :]
    y_test = y[401:, :]
    return (x_train, y_train), (x_test, y_test)


def draw_train_history(history):
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


def build_model():
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(13,)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer='adam',
                  loss='mean_squared_error')
    return model


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = load_data()

    model = build_model()
    early_stopping = EarlyStopping(monitor='loss', patience=10)
    history = model.fit(x_train, y_train,
                        epochs=500,
                        batch_size=64,
                        validation_split=0.2,
                        callbacks=[early_stopping])
    draw_train_history(history)

    loss = model.evaluate(x_test, y_test, batch_size=64)
    print("test loss: {}".format(loss))
```

### 模型结构

![](../../../.gitbook/assets/image%20%28419%29.png)

### 模型输出

```python
test loss: 0.20772670450664701
```

### 模型损失曲线

![](../../../.gitbook/assets/image%20%28416%29.png)

## keras集成学习-平均法

```python
import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import boston_housing
from keras.models import Model
from keras.layers import Input, Dense, average
from keras.callbacks import EarlyStopping

from sklearn.preprocessing import StandardScaler


def load_data():
    (x_train, y_train), (x_test, y_test) = boston_housing.load_data()
    x = np.vstack((x_train,x_test))
    y = np.concatenate((y_train, y_test))
    x = StandardScaler().fit_transform(x)
    y = StandardScaler().fit_transform(y.reshape(-1, 1))
    x_train = x[1:401, :]
    x_test = x[401:, :]
    y_train = y[1:401, :]
    y_test = y[401:, :]
    return (x_train, y_train), (x_test, y_test)


def draw_train_history(history):
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


def build_model():
    inputs = Input(shape=(13, ))
    model1_1 = Dense(64, activation='relu')(inputs)
    model2_1 = Dense(128, activation='relu')(inputs)
    model3_1 = Dense(32, activation='relu')(inputs)
    model1_2 = Dense(32, activation='relu')(model1_1)
    model2_2 = Dense(64, activation='relu')(model2_1)
    model3_2 = Dense(16, activation='relu')(model3_1)
    model1_out = Dense(1, activation='linear')(model1_2)
    model2_out = Dense(1, activation='linear')(model2_2)
    model3_out = Dense(1, activation='linear')(model3_2)
    out = average([model1_out, model2_out, model3_out])
    model = Model(inputs=inputs, outputs=out)
    model.compile(optimizer='adam',
                  loss='mean_squared_error')
    return model


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = load_data()

    model = build_model()
    early_stopping = EarlyStopping(monitor='loss', patience=10)
    history = model.fit(x_train, y_train,
                        epochs=500,
                        batch_size=64,
                        validation_split=0.2,
                        callbacks=[early_stopping])
    draw_train_history(history)
    model.save("regression-average-ensemble.h5")

    loss = model.evaluate(x_test, y_test, batch_size=64)
    print("test loss: {}".format(loss))
```

### 模型结构

![](../../../.gitbook/assets/image%20%28415%29.png)

### 模型输出

```python
test loss: 0.14132633038929532
```

### 模型损失曲线

![](../../../.gitbook/assets/image%20%28420%29.png)

## keras集成学习-学习法

```python
import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import boston_housing
from keras.models import Model
from keras.layers import Input, Dense, concatenate
from keras.callbacks import EarlyStopping

from sklearn.preprocessing import StandardScaler


def load_data():
    (x_train, y_train), (x_test, y_test) = boston_housing.load_data()
    x = np.vstack((x_train,x_test))
    y = np.concatenate((y_train, y_test))
    x = StandardScaler().fit_transform(x)
    y = StandardScaler().fit_transform(y.reshape(-1, 1))
    x_train = x[1:401, :]
    x_test = x[401:, :]
    y_train = y[1:401, :]
    y_test = y[401:, :]
    return (x_train, y_train), (x_test, y_test)


def draw_train_history(history):
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


def build_model():
    inputs = Input(shape=(13, ))
    model1_1 = Dense(64, activation='relu')(inputs)
    model2_1 = Dense(128, activation='relu')(inputs)
    model3_1 = Dense(32, activation='relu')(inputs)
    model1_2 = Dense(32, activation='relu')(model1_1)
    model2_2 = Dense(64, activation='relu')(model2_1)
    model3_2 = Dense(16, activation='relu')(model3_1)
    model1_3 = Dense(1, activation='linear')(model1_2)
    model2_3 = Dense(1, activation='linear')(model2_2)
    model3_3 = Dense(1, activation='linear')(model3_2)
    con = concatenate([model1_3, model2_3, model3_3])
    output = Dense(1, activation='linear')(con)
    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer='adam',
                  loss='mean_squared_error')
    return model

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = load_data()

    model = build_model()
    early_stopping = EarlyStopping(monitor='loss', patience=10)
    history = model.fit(x_train, y_train,
                        epochs=500,
                        batch_size=64,
                        validation_split=0.2,
                        callbacks=[early_stopping])
    draw_train_history(history)
    model.save("regression-ensemble.h5")

    loss = model.evaluate(x_test, y_test, batch_size=64)
    print("test loss: {}".format(loss))
```

### 模型结构

![](../../../.gitbook/assets/image%20%28417%29.png)

### 模型输出

```python
test loss: 0.16616634471075875
```

### 模型损失曲线

![](../../../.gitbook/assets/image%20%28418%29.png)

