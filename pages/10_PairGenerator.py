import streamlit as st
from aux.random_people_choice import random_people_choice

with st.form('teams_generator'):
    names = st.radio(
                'Choose a team', 
                [
                    'Дмитрий, Валентина, Иван, Андрей, Алексей, Диана, Санчай, Варвара',
                    'Гульназ, Пётр, Александр, Александра, Анна, Алексей, Андрей, Никита, Ольга'
                    # add here more names as str
                ]
            )
    team_names = st.text_input('Teams', placeholder='Logistic Regression, Gradient Boosting, Random Forest, Support Vector Machine')

    st.form_submit_button('Generate teams!', on_click=random_people_choice, args=(names.split(','), team_names.split(',')))   


if names and team_names:
    pairs = random_people_choice(names.split(), team_names.split(','))
    for team_name, names in pairs.items():
        st.write(f'{team_name}: {(" ".join(names))}')