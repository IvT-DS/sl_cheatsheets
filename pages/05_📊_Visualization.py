from random import seed
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
# import random_word
from random_word import Wordnik
wordnik_service = Wordnik()
from PIL import Image
from sklearn.linear_model import Ridge, Lasso, LinearRegression

# print(random_word.__version__)

# get words for xticks
# r = RandomWords()
# r.g
words = wordnik_service.get_random_words(limit=10)
# print(words)

# get prob distribution for values
np.random.seed(42)
x = np.random.poisson(lam=2, size=10)

if 'words' not in st.session_state:
    st.session_state['words'] = words

def softmax(x: np.array, temp: int = 1) -> np.array:
    """Returns softmax scores for input x

    Args:
        x (np.array): logits
        temp (int)  : temperature
    """    

    return np.exp(x/temp) / np.sum(np.exp(x/temp))

with st.expander('Softmax with temperature'):
    st.write(
        '''
            $$
                softmax(x)_i = \dfrac{e^{y_i / T}}{\sum_{j}^{N}e^{y_i / T}}
            $$
        '''
    )
    temp = st.slider('Temperature', 1, 30, 1, 1)
    left_col, right_col =  st.columns(2)
    
    with left_col:
        fig_1, ax_1 = plt.subplots()
        ax_1.bar(np.arange(len(x)), softmax(x), color='#4520AB')
        ax_1.set_title('Softmax')
        ax_1.set_xticks(np.arange(len(x)))
        ax_1.set_xticklabels(st.session_state['words'], rotation=90)
        st.pyplot(fig_1)
    with right_col:
        scores = softmax(x, temp=temp)
        fig_2, ax_2 = plt.subplots()
        ax_2.bar(np.arange(len(x)), scores, color='#4520AB')
        ax_2.set_title('Softmax with temp')
        ax_2.set_xticks(np.arange(len(x)))
        ax_2.set_xticklabels(st.session_state['words'], rotation=90)
        st.pyplot(fig_2)

with st.expander('Computation graph 🔥Pytorch'):
    st.image('https://miro.medium.com/max/504/0*4UHwQnsmUjyD7VtW.gif')

    st.markdown('1. [Computational graphs in PyTorch and TensorFlow](https://towardsdatascience.com/computational-graphs-in-pytorch-and-tensorflow-c25cc40bdcd1)')
    st.markdown('2. [Computation Graphs](https://www.cs.cornell.edu/courses/cs5740/2017sp/lectures/04-nn-compgraph.pdf)')

with st.expander('Singular value decomposition (SVD)'):
    st.write("""
    #### Сингулярное разложение матрицы на примере изображений
    Каждую черно-белую картинку размером M x N можно представить как матрицу размером M x N,
    где каждое значение в строке или столбце будет в диапазоне от 0 до 255 и будет обозначать
    степень градации серого от 0 - черный, до 255 - белый. 

    Что если нам нужно сократить объем хранимой информации, пусть и ценой потери качества? 
    При этом изображение должно оставаться узнаваемым.

    В этом может помочь [сингулярное разложение (SVD)](https://ru.wikipedia.org/wiki/%D0%A1%D0%B8%D0%BD%D0%B3%D1%83%D0%BB%D1%8F%D1%80%D0%BD%D0%BE%D0%B5_%D1%80%D0%B0%D0%B7%D0%BB%D0%BE%D0%B6%D0%B5%D0%BD%D0%B8%D0%B5)
    а точнее его главное практическое применение - возможность приблизить исходную матрицу 
    матрицей меньшего ранга.
    Разложив исходную матрицу изображения на три матрицы - U, Sigma и V мы можем взять 
    только первые k диагональных элементов (сингулярных значений) из матрицы Sigma, сохранив 
    при этом основную информацию об изображении. 

    Размер хранимой информации может сократиться очень существенно.

    Давайте попробуем!   
    """)
    st.caption('''
    *Для этого загрузите изображение :*
    ''')

    st.markdown("""
    Можно драг-н-дроп 
    """)


    uploaded_file = st.file_uploader(
        "Лучше ч/б, но и цветная не проблема - мы её обесцветим", 
        type=["jpg", "jpeg", "png"]
        )  
    
    if uploaded_file is not None:

        # Читаем с помощью PIL и сразу переводим в grayscale
        # без усреднения измерений
        img_arr = np.array(Image.open(uploaded_file).convert('L'))

        # делаем разложение сразу для выяснения max_k
        V, sing_values, U = np.linalg.svd(img_arr) 
        max_k = len(sing_values)

        # оформляем по-другому, чтобы не превышало максимальной длины строки по PEP
        k_components = st.slider(
            label='Количество сингулярных значений', 
            min_value=1, 
            max_value=len(sing_values), 
            value=50
        )

        # делаем две колонки для наглядности
        col1, col2 = st.columns(2)

        with col1:
            st.write('''
            Исходная картинка:
            ''')
            fig, ax = plt.subplots(1,1)
            ax.imshow(img_arr, cmap='gray')
            ax.axis('off')
            st.pyplot(fig)
        

        with col2:
            square_diagonal_sigma = np.diag(sing_values)
            num_col = U.shape[0] - square_diagonal_sigma.shape[1]
            num_col = int(num_col)
            sigma = np.hstack(
                (square_diagonal_sigma, np.zeros((square_diagonal_sigma.shape[0], num_col)))
            )
        
            st.write(
                k_components, '''сингулярных значений из''', max_k)

            V3,     = V[:, :k_components], 
            sigma3  = sigma[:k_components, :k_components], 
            U3      = U[:k_components, :]
            img_top = V3 @ sigma3 @ U3
        
            fig_result, ax_result = plt.subplots(1,1)
            ax_result.imshow(img_top[0], cmap='gray')
            ax_result.axis('off')
            st.pyplot(fig_result)

        st.write('''#### Размер исходных матриц:''')
        st.write('V = ', V.shape[0], 'x', V.shape[1], '=', V.shape[0]*V.shape[1], 'значений'  )
        st.write('Sigma = ', sigma.shape[0], 'x', sigma.shape[1], '=', sigma.shape[0] * sigma.shape[1], 'значений'  )
        st.write('U = ', U.shape[0], 'x', U.shape[1], '=', U.shape[0] * U.shape[1], 'значений'  )
        st.write('''#### Размер новых матриц:''')
        st.write('V = ', V3.shape[0], 'x', V3.shape[1], '=', V3.shape[0]*V3.shape[1], 'значений'  )
        st.write('Sigma = ', sigma3[0].shape[0], 'x', sigma3[0].shape[1], '=', sigma3[0].shape[0] * sigma3[0].shape[1], 'значений'  )
        st.write('U = ', U3.shape[0], 'x', U3.shape[1], '=', U3.shape[0] * U3.shape[1], 'значений'  )
        
        
    st.markdown('[@trojanof](https://github.com/trojanof)')


with st.expander('Регуляризация'): 

    alpha = st.slider('Alpha', 0, 20, value=1)

    x = np.linspace(0, 10, 100)
    x_outliers = np.linspace(1.5, 3, 20)
    y = x*1.5 + np.random.normal(0, 2, size=100)
    y_outliers = x_outliers * 7 + np.random.normal(0, 3, size=x_outliers.shape[0])

    x = np.concatenate((x, x_outliers))
    y = np.concatenate((y, y_outliers))

    ols = LinearRegression()
    ridge = Ridge(alpha=alpha, max_iter=10000)
    lasso = Lasso(alpha=alpha, max_iter=10000)

    ols.fit(x.reshape(-1, 1), y.reshape(-1, 1))
    ridge.fit(x.reshape(-1, 1), y.reshape(-1, 1))
    lasso.fit(x.reshape(-1, 1), y.reshape(-1, 1))

    

    fig, ax = plt.subplots()
    ax.scatter(x, y, marker='.')
    ax.plot(np.arange(0, 11), ols.predict(np.arange(11).reshape(-1, 1)), c='cyan', label='OLS')
    ax.plot(np.arange(0, 11), ridge.predict(np.arange(11).reshape(-1, 1)), c='red', label='Ridge')
    ax.plot(np.arange(0, 11), lasso.predict(np.arange(11).reshape(-1, 1)), c='green', label='Lasso')
    plt.legend()
    st.pyplot(fig)