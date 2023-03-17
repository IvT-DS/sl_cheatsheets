import streamlit as st
import streamlit.components.v1 as components


st.header('Функции потерь, активации и число нейронов выходного слоя')

st.markdown(
    '''
|     | Задача                       | Функция потерь                                                                         | Число выходных нейронов |
| --- | ---------------------------- | -------------------------------------------------------------------------------------- | ----------------------- |
| 1   | Бинарная классификация       | Бинарная кросс-энтропия  `BCELoss()`                                                  | 1                       |
| 2   | Бинарная классификация       | Бинарная кросс-энтропия __без активации последнего нейрона__ ->  `BCEWithLogitsLoss()` | 1                       |
| 3   | Бинарная классификация       | Категориальная кросс-энтропия  `CrossEntropyLoss()`                                   | 2                       |
| 4   | Многоклассовая классификация | Категориальная кросс-энтропия  `CrossEntropyLoss()`                                   | Число классов           |
| 5   | Регрессия                    | Любая регрессионная  `MSELoss(), L1Loss()`, etc                                       | 1                       |
''')

st.markdown('''

#### Случай № 1
```python

model = nn.Sequential(
    nn.Linear(n, m),
    nn.Sigmoid(),
    nn.Dropout(),
    nn.Linear(m, 1),
    nn.Sigmoid()
)

predictions = model(X) # числа в интервале [0; 1]

loss = torch.nn.BCELoss(predictions, target)
loss.backward()
```

#### Случай № 2

Часто для оптимизации вычислений в задаче бинарной классификации не активируют выходной слой сигмоидой, тогда необходимо использовать в качестве функции потерь `torch.nn.BCEWithLogitsLoss`, однако для получения распределений __вероятностей__ принадлежности объекта классу все равно нужно будет применять сигмоиду. 

```python


model = nn.Sequential(
    nn.Linear(n, m),
    nn.Sigmoid(),
    nn.Dropout(),
    nn.Linear(m, 1)
)

predictions = model(X) 
loss = torch.nn.BCEWithLogitsLoss(predictions, target)
loss.backward()

# получаем вероятности принадлежности объекта классу
probabilities = torch.functional.sigmoid(predictions) # числа в интервале [0; 1]

```
#### Случай № 4
При решении задачи многоклассовой классификации функцию софтмакса мы применяем __только__ для того, чтобы получить вероятностное распределение между классами. В функцию потерь мы передаём «сырые» значения с выходного слоя (т.е. не активированные) – __логиты__. Функция `torch.nn.CrossEntropyLoss()` ожидает на вход именно их, а не числа в интервале $$[0; 1]$$. 


```python
# K - число классов
model = nn.Sequential(
    nn.Linear(n, m),
    nn.Sigmoid(),
    nn.Dropout(),
    nn.Linear(m, K)
)

predictions = model(X) # ⬅️ логиты – любые числа в интервале [-∞; +∞]
loss = torch.nn.CrossEntropyLoss(predictions, target)
loss.backward()

# получаем вероятности принадлежности объекта классу
probabilities = torch.functional.softmax(predictions) # числа в интервале [0; 1]

```
''')


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