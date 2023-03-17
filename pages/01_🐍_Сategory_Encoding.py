import streamlit as st
import pandas as pd
import streamlit.components.v1 as components


st.title('Кодирование категориальных признаков')

'''Много данных имеют категориальную природу: 
* уровень образования (начальный / средний / высший) 
* округ (САО / ВАО / ЦАО) 
* индекс 
* и т.д.
'''


'## Зачем'

'Модели не умеют работать с такими данными напрямую, их нужно представить в числовом виде.'

'## Способы'

'''

`sklearn`
* [Label Encoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html)
     * [Ordinal Encoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OrdinalEncoder.html)
 * [One Hot Encoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html) 

 `target encoder`
 * [Binary Encoder](https://contrib.scikit-learn.org/category_encoders/binary.html)
 * [Helmert Encoder](https://contrib.scikit-learn.org/category_encoders/helmert.html)
 * [Backward-Difference Encoder](https://contrib.scikit-learn.org/category_encoders/backward_difference.html)
 * [Target Encoder](https://contrib.scikit-learn.org/category_encoders/targetencoder.html)
'''


"""---"""
'### Label Encoder'

'Кодируем каждое уникальное значение признака цифрой от $0$ до $N-1$, где $N$ - число уникальных значений признака'

'''
```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
le.fit(df['category'])
encoded_district = le.transform(df['category'])
df.assign(encoded=encoded_district)
```

|    | category   |   encoded | 
|---:|:-----------|----------:| 
|  0 | VAO        |         2 |
|  1 | CAO        |         0 |
|  2 | SAO        |         1 |


🟢 Простой 

🔴 Искажает данные: теперь `VAO` стало в два раза «больше», чем `SAO`

❗️ Принимает на вход только вектор размера `(n_samples)`, если нужно закодировать несколько столбцов сразу, то следует использовать `OrdinalEncoder` 

❗️ Если данные в порядковой шкале, то нужно явно задать порядок: `{низкий : 0, средний : 1, высокий : 2}` 

🐍 [sklearn.preprocessing.LabelEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html) 

🐍 [sklearn.preprocessing.OrdinalEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OrdinalEncoder.html#sklearn.preprocessing.OrdinalEncoder)
'''

"""---"""

'### One Hot Encoder' 

'''
Создаем столько же новых столбцов, сколько уникальных значений признака. 
На пересечении строки и столбца 1, если объект является носителем признака, 0 – если нет''' 

'''
```python   
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse=False)
ohe.fit_transform(df)
```

|    | category   |   CAO |   SAO |   VAO |
|---:|:-----------|-----------:|-----------:|-----------:|
|  0 | VAO        |          0 |          0 |          1 |
|  1 | CAO        |          1 |          0 |          0 |
|  2 | SAO        |          0 |          1 |          0 |

🟢 Хорошо интерпретируется 

🔴 Сильно увеличивает пространство признаков

🐍 [sklearn.preprocessing.OneHotEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html#sklearn.preprocessing.OneHotEncoder)

🐍 [padnas.get_dummies](https://pandas.pydata.org/docs/reference/api/pandas.get_dummies.html)

'''

"""---"""

'### Binary Encoder'

'''

Принцип в том, что десятичное число $N$ можно представить $\log_2N$ бинарными значениями. 
Например число $22$ можно представить как $10110$, т.е $5$ битами.
'''

'''
```python
from category_encoders import BinaryEncoder
be = BinaryEncoder()
pd.concat([df, be.fit_transform(df)], axis=1)
```


|    | category   |   category_0 |   category_1 |   category_2 |
|---:|:-----------|:------------:|-------------:|-------------:|
|  0 | VAO        |            0 |            0 |            1 |
|  1 | CAO        |            0 |            1 |            0 |
|  2 | SAO        |            0 |            1 |            1 |
|  3 | SVAO       |            1 |            0 |            0 |
|  4 | SZAO       |            1 |            0 |            1 |



🟢 Относительно компактный  

🔴 Пропадает интерпретируемость

🐍 [BinaryEncoder](https://contrib.scikit-learn.org/category_encoders/binary.html)
'''
"""---"""

'### Helmert Encoder'

'''
Создается матрица размера $(N, N–1)$, выше главной диагонали и на ней расположены $–1$, под главной диагональю порядковые номера признаков, всё остальное – $0$

```python
from category_encoders.helmert import HelmertEncoder
he = HelmertEncoder(drop_invariant=True)
pd.concat([df, he.fit_transform(df)], axis=1) 
```

|    | category   |   category_0 |   category_1 |   category_2 |
|---:|:-----------|-------------:|-------------:|-------------:|
|  0 | VAO        |           -1 |           -1 |           -1 |
|  1 | CAO        |            1 |           -1 |           -1 |
|  2 | SAO        |            0 |            2 |           -1 |
|  3 | SVAO       |            0 |            0 |            3 |


🟢 Подходит для порядковых переменных 

🔴 Неочевиден в интерпретации

❗️ Не единственная интерпретация: [How to calculate Helmert Coding](https://stats.stackexchange.com/questions/411134/how-to-calculate-helmert-coding/411837#411837)

🐍 [Helmert Encoder ](https://contrib.scikit-learn.org/category_encoders/helmert.html)

'''

"""---"""

'''
### Backward-Difference Encoder

Похож на `Helmert Encoder`. 
'''

st.image('aux/bd_encoder.png', width=280)
st.caption('$k$ – число уникальных значений признака')

'''
```python
from category_encoders import BackwardDifferenceEncoder
bde = BackwardDifferenceEncoder(drop_invariant=True)
pd.concat([df, 
           bde.fit_transform(df)], 
          axis=1)
```

|    | category   |   category_0 |   category_1 |   category_2 |   category_3 |
|---:|:-----------|-------------:|-------------:|-------------:|-------------:|
|  0 | VAO        |         -0.8 |         -0.6 |         -0.4 |         -0.2 |
|  1 | CAO        |          0.2 |         -0.6 |         -0.4 |         -0.2 |
|  2 | SAO        |          0.2 |          0.4 |         -0.4 |         -0.2 |
|  3 | SVAO       |          0.2 |          0.4 |          0.6 |         -0.2 |
|  4 | SZAO       |          0.2 |          0.4 |          0.6 |          0.8 |\


🐍 [Backward Difference Coding](https://contrib.scikit-learn.org/category_encoders/backward_difference.html)

'''



"""---"""
'''
### Target Encoder

Заменяет категорию на ([сглаженное](https://www.kaggle.com/code/ryanholbrook/target-encoding/tutorial)) среднее по целевому признаку: 

 ```python
from category_encoders import TargetEncoder
te = TargetEncoder()
pd.concat([df, te.fit_transform(df['category'], df['target'])], axis=1)
```

#### Классификация


|    | category   |   target |   category |
|---:|:-----------|---------:|-----------:|
|  0 | VAO        |        1 |   0.526894 |
|  1 | VAO        |        0 |   0.526894 |
|  2 | SAO        |        0 |   0.65872  |
|  3 | SAO        |        1 |   0.65872  |
|  4 | SAO        |        1 |   0.65872  |


#### Регрессия

|    | category   |   target |   category |
|---:|:-----------|---------:|-----------:|
|  0 | VAO        |      120 |    68.6038 |
|  1 | VAO        |       10 |    68.6038 |
|  2 | SAO        |       12 |    86.2685 |
|  3 | SAO        |      100 |    86.2685 |
|  4 | SAO        |      150 |    86.2685 |


🟢 Имеет статистическое обоснование 

🟢 Компактный 

🔴 Наличие выбросов в целевых переменных может «смещать» оценки 

🔴 Требует наличия целевой переменной 

🐍 [Target Encoder](https://contrib.scikit-learn.org/category_encoders/targetencoder.html)

'''

components.html(
    """
    <!-- Yandex.Metrika counter -->
<script type="text/javascript" >
   (function(m,e,t,r,i,k,a){m[i]=m[i]||function(){(m[i].a=m[i].a||[]).push(arguments)};
   m[i].l=1*new Date();
   for (var j = 0; j < document.scripts.length; j++) {if (document.scripts[j].src === r) { return; }}
   k=e.createElement(t),a=e.getElementsByTagName(t)[0],k.async=1,k.src=r,a.parentNode.insertBefore(k,a)})
   (window, document, "script", "https://mc.yandex.ru/metrika/tag.js", "ym");

   ym(92504528, "init", {
        clickmap:true,
        trackLinks:true,
        accurateTrackBounce:true,
        webvisor:true
   });
</script>
<noscript><div><img src="https://mc.yandex.ru/watch/92504528" style="position:absolute; left:-9999px;" alt="" /></div></noscript>
<!-- /Yandex.Metrika counter -->
""")