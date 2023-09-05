import streamlit as st

import streamlit.components.v1 as components

from sklearn.metrics import roc_curve
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.header('Классификация')


st.image('https://miro.medium.com/max/1273/1*vbfuOkRGdcNkUXp8uEi4BA.png', width=300)

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
    st.write('Метрика высока, однако модель не обнаружила ни одного объекта класса 1. ')

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

np.random.seed(42)
noise = np.random.normal(loc=0.01, scale=.03, size=80)

# divider = st.slider(
#     'divider', min_value=1., max_value=3., step=.5
# )

good = [0.8, .9, .8, .9, .7, .6, .8, .9, .2, .3]*4 + [.2, .14, .4, .61, .18, .4, .5, .4, .2, .2]*4
good = (good + noise)
true = [1]*40 + [0]*40




fpr, tpr, t = roc_curve(true, good)
tre = st.slider(
    'Порог отнесения к классу 1', 
    min_value=0., 
    max_value=1., 
    step=.02,
    )
# st.write(t)
# tre = st.select_slider(
#     'Порог отнесения к классу 1', 
#     options=[1] + [str(i) for i in t[1:]]
# )

index_of_pos = sum(t < float(tre)-.001)
st.latex(' \\text{True positive rate}=\dfrac{TP}{TP+FN}, \quad \\text{False Positive Rate} = \dfrac{FP}{FP+TN}')

left_col, right_col =  st.columns(2)
# with left_col:
fig_1, ax_1 = plt.subplots(figsize=(12, 3))
sns.kdeplot(good[:10], ax=ax_1, label='Положительный класс', c='green')
sns.kdeplot(good[-10:], ax=ax_1, label='Отрицательный класс', c='red')
plt.axvline(tre, ymin=0, ymax=2, linestyle='--', c='gray')
plt.fill_between(np.linspace(tre, 1.2, 10), [0]*10, [2.65]*10, color='green', alpha=.12)
plt.fill_between(np.linspace(0, tre, 10), [0]*10, [2.65]*10, color='red', alpha=.12)
ax_1.set_xlim(0, 1.2)
ax_1.set_ylim(0, 2.6)
ax_1.set_xlabel('Распределение вероятностей предсказаний для классов 1 и 0')
ax_1.legend()
st.pyplot(fig_1)
# with right_col:
    # fig_2, ax_2 = plt.subplots()
    # ax_2.plot(fpr, tpr, marker='.')
    # ax_2.scatter(fpr[index_of_pos], tpr[index_of_pos], c='red', marker='o', s=40*3)
    # ax_2.set_ylabel('True positive rate')
    # ax_2.set_xlabel('False positive rate')
    # ax_2.text(.25, .1, f'TPR={tpr[index_of_pos]}, FPR={fpr[index_of_pos]}', fontsize=20)
    # st.pyplot(fig_2)

pair_fprtpr = pd.DataFrame(
{
    'fpr': fpr,
    'tpr': tpr
}
)
fig = px.line(pair_fprtpr,  x='fpr', y='tpr', markers=True)
fig.add_trace(
    go.Scatter(
        x=[fpr[index_of_pos]], 
        y=[tpr[index_of_pos]],
        mode='markers',
        hovertext=f'Threshold: {t[index_of_pos]:1f}', name=""))
fig.update_xaxes(range=(-.01, 1.1))
fig.update_layout(showlegend=False)
fig.add_annotation(
    dict(font=dict(size=25),
        x=.8, y=.1,
        text=f'TPR={tpr[index_of_pos]}, FPR={fpr[index_of_pos]}',
        showarrow=False
    )
)
st.plotly_chart(fig, use_container_width=False)

st.dataframe(pd.DataFrame(
    {
        'True labels' : true, 
        'Predictions' : good
    }
).T)










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
