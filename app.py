import streamlit as st
import os
from langchain.llms import OpenAI
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain , SequentialChain


load_dotenv()
api_key = os.getenv("API_KEY")


llm = OpenAI(temperature=0.7,)


PromptTemplate_1 = PromptTemplate(input_variables=['problem'],
                                 template= """You are an expert mental health advisor, you have to behave friendly with people.
                                                your task is to help people who are going through stress or mental health problem,
                                                they will describe their problem and you have to provide useful answers,
                                                 you can suggest them songs, meditation, yoga or any other useful things.
                                                 write it within 100 words. DO NOT ASK ANY PERSONAL INFORMATION. DO NOT ANSWER ANY QUESTION BEYOND MENTAL HEALTH.
                                                 
                                                  
                                                user problem : {problem} """)

mental_health = LLMChain(llm=llm , prompt=PromptTemplate_1,output_key='health')




PromptTemplate_2 = PromptTemplate(input_variables=['health'],
                                 template= """You are an expert mental health advisor, you have to behave friendly with people.
                                                your task is provide which type of meditations and yogas the user can do.
                                                dont say hi hello, just suggest yogas and meditations.
                                                DO NOT ASK ANY PERSONAL INFORMATION. DO NOT ANSWER ANY QUESTION BEYOND MENTAL HEALTH.
                                                write it within 150 words.
                                                
                                                based on my probblem : {health} , suggest me yogas i can do.
                                                format like this:
                                                1.
                                                2.
                                                3. """)
mental_health_2 = LLMChain(llm=llm , prompt=PromptTemplate_2,output_key='yoga',)




seq_chain = SequentialChain(chains=[mental_health,mental_health_2],input_variables=['problem'],output_variables=['health','yoga'])




st.title("Mental Health Advisor LLM 🦜️")
st.subheader("Nurturing Minds: Your Compassionate Companion on the Mental Health Journey")
input_ = st.text_input(label="Enter your query",max_chars=100)
button = st.button('Submit',)

if button:
    query = input_.title()
    if query != '':
        st.success('Got your query, Processing !!!')
        output = seq_chain({'problem':query})
       
        st.markdown(f':red[**User Query :**] *{output["problem"]}*')
        st.markdown(f':green[**AI :**] {output["health"]}')
        st.markdown(f":green[**Detailed  Suggestions:**]")
        st.markdown(f'### {output["yoga"]}')
        
    else:
        st.warning('Query is blank..')
    