import streamlit as st

import streamlit.components.v1 as components

st.header('Классификация')


st.image('https://miro.medium.com/max/1273/1*vbfuOkRGdcNkUXp8uEi4BA.png')

'''
- **TP** • True Positive • Истинно-положительный: классификатор отнес объект к классу `1` и объект действительно относится к классу `1` 
- **FP** • False Positive • Ложно-положительный: классификатор отнес объект к классу `1`, но на самом деле объект относится к классу `0`
- **FN** • False Negative • Ложно-отрицательный: классификатор отнес объект к классу `0`, но на самом деле объект относится к классу `1` 
- **TN** • True Negative • Истинно-отрицательный: классификатор отнес объект к классу `0`, и объект действительно относится к классу `0`

## Accuracy
'''
st.info('Доля правильных ответов')
st.latex('''acc = \dfrac{TP + TN}{TP + FP + FN + TN}''')
st.latex('''acc \in [0; 1]''')

'''
Если выборка несбалансирована, то эта метрика ничего не покажет.
'''

with st.expander('Пример'):
    st.write('Выборка состоит из 10 объектов класса `1` и 90 объектов класса `0`. Классификатор выдает константный прогноз: любой объект относится к классу `0`. Тогда')
    st.latex('''
    acc = \dfrac{90}{90+10} = 0.9
    '''
    )
    st.write('Метрика высока, однако модель не обнаружила ни одного объекта класса 0. ')

'''
## Precision • Точность
'''
st.info('Доля правильных позитивных ответов среди всех названных позитивными. Тем выше, чем ниже число ложно-положительных срабатываний. ')

st.latex(''' precision = \dfrac{TP}{TP+FP} ''')

'''
## Recall • Полнота
'''
st.info('Доля правильных позитивных ответов среди всех названных позитивными. Тем выше, чем ниже число ложно-отрицательных срабатываний. ')

st.latex(''' recall = \dfrac{TP}{TP+FN} ''')

'''
## F-score  
'''
st.info('Гармоническое среднее между точностью и полнотой. Объединяет две метрики в одну. В базовом варианте точность и полнота имеют одинаковый вес. ')

st.latex('''f = 2 × \dfrac{precision × recall}{precision+recall}''')

'''
### F-beta score
'''
st.info('Модификация f1-score для придания разным компонентам разного веса (вклада) в итоговую метрику. ')

st.latex('''f_{β} = (1+β^2) \dfrac{precision × recall}{β^2 × precision+recall}''')



'''
### Macro- и micro- варианты
* Часто метрики `precision`, `recall` и `f1` можно встретить с приставкой `micro` или `macro`
* Посмотрим на примере `precision`
'''

with st.expander('Пусть есть confusion matrix для классификатора:'):

    st.image('https://i.stack.imgur.com/tcylh.png')
    st.caption('Строкам соответствуют ответы классификатора (system output), столбцы – настоящие лейблы. `Precision` и `recall` вычисляются в разрезе одного класса. ')

    '''
    Тогда `micro-` и `macro-` варианты для `precision` будут вычисляться так: 
    '''

    st.image('https://i.stack.imgur.com/Nh4Yl.png')
    st.caption('Macro вариант нормирует на число классов, micro – на объем выборки, для которого имеются предсказания. ')

'''
## ROC-AUC
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
