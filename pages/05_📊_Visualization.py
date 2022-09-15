from random import seed
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from random_word import RandomWords
r = RandomWords()

np.random.seed(42)
x = np.random.poisson(lam=2, size=10)

def softmax(x: np.array, temp: int = 1) -> np.array:
    """Returns softmax scores for input x

    Args:
        x (np.array): logits
        temp (int)  : temperature
    """    

    return np.exp(x/temp) / np.sum(np.exp(x/temp))

with st.expander('Softmax with temperature'):
    temp = st.slider('Temperature', 1, 100, 1, 5)
    left_col, right_col =  st.columns(2)
    words = r.get_random_words(limit=10)
    with left_col:
        fig_1, ax_1 = plt.subplots()
        ax_1.bar(np.arange(len(x)), softmax(x))
        ax_1.set_title('Softmax')
        ax_1.set_xticks(np.arange(len(x)))
        # ax_1.set_xticklabels(words, rotation=90)
        print(words)
        st.pyplot(fig_1)
    with right_col:
        
        scores = softmax(x, temp=temp)
        fig_2, ax_2 = plt.subplots()
        ax_2.bar(np.arange(len(x)), scores)
        ax_2.set_title('Softmax with temp')
        ax_2.set_xticks(np.arange(len(x)))
        # ax_2.set_xticklabels(words, rotation=90)
        st.pyplot(fig_2)

    