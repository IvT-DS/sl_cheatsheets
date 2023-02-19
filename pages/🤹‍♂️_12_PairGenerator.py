import streamlit as st
import numpy as np
from aux.random_people_choice import random_people_choice

teams = np.array([
    'Pandas', 'Matplotlib', 'Seaborn',
    'Poisson', 'Bernoulli', 'Gauss',
    'XGBoost', 'LightGBM', 'CatBoost',
    'Dropout', 'Convolution','Linear',
    'GPT', 'BERT', 'LSTM', 'RNN', 
    'ResNet', 'Inception', 'DenseNet', 
    'YOLO', 'FasterRCNN'
])

st.sidebar.write('Select Filter')

def checks():
    labels = [
        st.sidebar.checkbox('pandas'),  st.sidebar.checkbox('matplotlib'), 
        st.sidebar.checkbox('seaborn'),         st.sidebar.checkbox('Poisson'), 
        st.sidebar.checkbox('Bernoulli'),       st.sidebar.checkbox('Gauss'),
        st.sidebar.checkbox('XGBoost'), st.sidebar.checkbox('LightGBM'), 
        st.sidebar.checkbox('CatBoost'),        st.sidebar.checkbox('Dropout'), 
        st.sidebar.checkbox('Convolution'),     st.sidebar.checkbox('Linear'),
        st.sidebar.checkbox('GPT'),     st.sidebar.checkbox('BERT'), 
        st.sidebar.checkbox('LSTM'),            st.sidebar.checkbox('RNN'), 
        st.sidebar.checkbox('ResNet'),  st.sidebar.checkbox('Inception'), 
        st.sidebar.checkbox('DenseNet'),        st.sidebar.checkbox('YOLO'), 
        st.sidebar.checkbox('FasterRCNN')
    ]

    return labels

labels = teams[checks()]

# with st.form('teams_generator'):
names = st.radio(
            ' ', 
            [
                # 'Дмитрий, Валентина, Иван, Андрей, Алексей, Диана, Санчай, Варвара',
                # 'Гульназ, Пётр, Александр, Александра, Анна, Никита, Ольга'
                'Сиражудин, Мила, Семен, Анатолий, Гор, Матвей', 
                'Никита, Сева, Костя, Катя, Ваня, Рома'
                # add here more names as str
            ]
        )
    # team_names = st.text_input('Teams', placeholder='Convolutional, Dropout, Linear, BatchNormalization')

    # st.form_submit_button('Generate teams!', on_click=random_people_choice, args=(names.split(','), labels))   


# print(labels)
if names and len(labels) != 0:
    pairs = random_people_choice(names.split(','), labels)
    for team_name, names in pairs.items():
        st.markdown(f'__{team_name}__:  {(", ".join(names))}')