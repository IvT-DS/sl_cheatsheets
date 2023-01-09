from random import seed
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
# import random_word
from random_word import Wordnik
wordnik_service = Wordnik()

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
        ax_1.bar(np.arange(len(x)), softmax(x))
        ax_1.set_title('Softmax')
        ax_1.set_xticks(np.arange(len(x)))
        ax_1.set_xticklabels(st.session_state['words'], rotation=90)
        st.pyplot(fig_1)
    with right_col:
        scores = softmax(x, temp=temp)
        fig_2, ax_2 = plt.subplots()
        ax_2.bar(np.arange(len(x)), scores)
        ax_2.set_title('Softmax with temp')
        ax_2.set_xticks(np.arange(len(x)))
        ax_2.set_xticklabels(st.session_state['words'], rotation=90)
        st.pyplot(fig_2)

with st.expander('Computation graph 🔥Pytorch'):
    st.image('https://miro.medium.com/max/504/0*4UHwQnsmUjyD7VtW.gif')