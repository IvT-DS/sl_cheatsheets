import streamlit as st

import streamlit.components.v1 as components


st.header('Регрессия')

st.write('''$y_i$ - истинное значение на $i$-ом объекте''')
st.write('''$\hat{y}_i$ - предсказанное значение на $i$-ом объекте''')

'''
### Mean squared error • MSE
'''

st.info('MSE применяется в ситуациях, когда нам надо подчеркнуть большие ошибки и выбрать модель, которая дает меньше больших ошибок прогноза.')

st.latex('''MSE = \dfrac{1}{N} \sum_{i=1}^{N}(y_i - \hat{y}_i)^2''')

'''
Обоснована статистически: берётся из метода максимального правдоподобия. 
'''

'''
### Root Mean squared error • RMSE
'''

st.info('Легко интерпретировать, поскольку он имеет те же единицы, что и исходные значения (в отличие от MSE). ')

st.latex('''RMSE = \sqrt{\dfrac{1}{N} \sum_{i=1}^{N}(y_i - \hat{y}_i)^2}''')

'''
### Mean Absolute Error • MAE
'''

st.info('Среднеквадратичный функционал сильнее штрафует за большие отклонения по сравнению со среднеабсолютным, и поэтому более чувствителен к выбросам.')

st.latex('''MAE =\dfrac{1}{N} \sum_{i=1}^{N}|y_i - \hat{y}_i|''')

'''
### Mean Absolute Percentage Error • MAPE
'''

st.info('Этот коэффициент можно интерпретировать в долях или процентах. Если  получилось, например, что MAPE=11.4%, то это говорит о том, что ошибка составила 11,4% от фактических значений. ')

st.latex('''MAPE = \dfrac{1}{N} \sum_{i=1}^{N} | \dfrac{y_i - \hat{y}_i}{y_i} | × 100''')

'''
### Symmetric mean absolute percentage error • SMAPE
'''

st.info('Может вычисляться по-разному: 📝[Symmetric mean absolute percentage error](https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error) ')

st.latex('''SMAPE = \dfrac{100\%}{N} \sum_{i=1}^{N} \dfrac{|\hat{y}_i - y_i|}{|y_i| + |\hat{y}_i|}''')

'''
### R2-score
'''

st.info('Доля объясненной дисперсии. «Насколько лучше мы предсказываем по сравнению с тем, если бы предсказывали среднее значение для всех целевых объектов»')

st.latex('''R^2  = 1 - \dfrac{\sum_{i=1}^{N} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{N} (y_i - \overline{y})^2}''')

'''
### Adjusted R2-score
'''

st.info('Скорректированный на объем выборки и число предикторов R2-score.')

st.latex('''R^2_{adj} =  1 - [ \dfrac{(1-R^2)(N-1)}{(N-k-1)} ]''')
st.caption('$N$ - объем выборки, $k$ – число предикторов')

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