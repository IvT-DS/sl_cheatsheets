import streamlit as st
from aux.random_people_choice import random_people_choice

with st.form('teams_generator'):
    names = st.radio(
                'Choose a team', 
                [
                    # 'Дмитрий, Валентина, Иван, Андрей, Алексей, Диана, Санчай, Варвара',
                    # 'Гульназ, Пётр, Александр, Александра, Анна, Никита, Ольга'
                    'Сиражудин, Мила, Семен, Анатолий, Гор, Матвей', 
                    'Никита, Сева, Костя, Катя, Ваня, Рома'
                    # add here more names as str
                ]
            )
    team_names = st.text_input('Teams', placeholder='Convolutional, Dropout, Linear, BatchNormalization')

    st.form_submit_button('Generate teams!', on_click=random_people_choice, args=(names.split(','), team_names.split(',')))   


if names and team_names:
    pairs = random_people_choice(names.split(','), team_names.split(','))
    for team_name, names in pairs.items():
        st.markdown(f'{team_name} {(", ".join(names))}')