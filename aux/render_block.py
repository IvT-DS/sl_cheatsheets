import streamlit as st

class Block(object):
    '''
    question: str, question text
    probable_location: str, where can be found
    error_text: str, pytorch error
    text_solution: str, human readable wording
    code_solution: str, step(s) to fix
    '''
    def __init__(
        self, 
        # question:str, 
        probable_location:str, 
        error_text:str, 
        text_solution:str, 
        code_solution:str
    ):
        # self.question = '##### ' + question
        self.probable_location = '🔎 ' + probable_location
        self.error_text = '```' + error_text + '```'
        self.text_solution = text_solution
        self.code_solution = code_solution

    def render_block(self): 
        with st.container():
            # st.markdown(self.question)
            st.error(self.error_text)
            st.write(self.probable_location)
            with st.expander('✅ Решение'):
                st.write(self.text_solution)
                st.code(self.code_solution)
        st.write("""---""")
            