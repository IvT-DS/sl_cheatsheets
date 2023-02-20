import streamlit as st
from aux.random_people_choice import random_people_choice

st.header('Генератор команд')

teams = st.multiselect('Названия команд', [
    'pandas', 'matplotlib', 'seaborn',
    'Poisson', 'Bernoulli', 'Gauss',
    'XGBoost', 'LightGBM', 'CatBoost',
    'Dropout', 'Convolution','Linear',
    'GPT', 'BERT', 'LSTM', 'RNN', 
    'ResNet', 'Inception', 'DenseNet', 
    'YOLO', 'FasterRCNN'
])

names = st.radio(
            ' ', 
            [
                'Сиражудин, Мила, Семен, Анатолий, Гор, Матвей', 
                'Никита, Сева, Костя, Катя, Ваня, Рома',
                'Алексей, Аня, Галина, Владислав, Егор, Елена, Иван, Никита'
                # add here more names as str
            ]
        )


# print(labels)
st.markdown('---------')
if names and len(teams) != 0:
    pairs = random_people_choice(names.split(','), teams)
    for team_name, names in pairs.items():
        st.markdown(f'__{team_name}__:  {(", ".join(names))}')