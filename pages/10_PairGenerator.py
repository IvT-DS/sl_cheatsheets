import streamlit as st
from numpy.random import choice
from aux.random_people_choice import random_people_choice

with st.form('teams_generator'):

    # names = st.text_input('Names', placeholder='Julia John Catherine')
    names = st.radio(
                'Choose a team', 
                [
                    'Дмитрий, Валентина, Иван, Андрей, Алексей, Диана, Санчай, Варвара',
                    'Гульназ, Пётр, Александр, Александра, Анна, Алексей, Андрей, Никита, Ольга'
                ]
            )
    team_names = st.text_input('Teams', placeholder='   ')

    st.form_submit_button('Generate teams!', on_click=random_people_choice, args=(names.split(','), team_names.split(',')))   

# st.write(st.session_state['teams'])

if names and team_names:
    pairs = random_people_choice(names.split(), team_names.split(','))
    for team_name, names in pairs.items():
        st.write(f'{team_name}: {(" ".join(names))}')
        

# people = [
#     'Дмитрий', 'Валентина', 
#     'Иван','Андрей',
#     'Алексей','Диана',
#     'Санчай', 'Варвара']