import streamlit as st
import torch
from torch import nn

'''
# Подготовка данных для моделей `PyTorch`

Для подгтовки данных в PyTorch используются специальные классы. Если данных мало, 
то можно просто конвертировать их в тензор и подавать на вход модели, но если необходимо разбивать данные на 
батчи, то нужно использовать `DataLoader`. 

## Табличные данные

### Простой вариант (не рекомендуется)

```
from sklearn.datasets import make_classification
X, y = make_classification()
print(f'Types: {type(X)}, {type(y)}')

Types: <class 'numpy.ndarray'>, <class 'numpy.ndarray'>

model = nn.Sequential(
    nn.Linear(20, 32),
    nn.Sigmoid(),
    nn.Linear(32, 1)
)

model(X)

TypeError: linear(): argument 'input' (position 1) must be Tensor, not numpy.ndarray
``` 

Конвертировать данные в тензоры можно так: 

```
X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)

y_pred = model(X)
print(y_pred)

tensor([[0.0214],
        [0.0659],
        [0.1001]], grad_fn=<SliceBackward0>)
```
'''
