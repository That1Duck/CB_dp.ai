import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(os.getcwd())/'.env')

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI


llm = ChatOpenAI(model = "gpt-4o-mini", temperature= 0)
prompt = ChatPromptTemplate.from_template("""
You work as an intelligent assistant. You receive a user request. 
Determine which is best suited: “course” or “blog.” 
If the user wants to **learn**, to study a **new skill**, it is better to choose “course.”
If they just want to **read** or **get an overview**, choose “blog.”

Answer with **one word**: “course” or “blog.”

Request: {user_input}

""")

chain = prompt | llm | StrOutputParser()

def decide_type(user_input):
    result = chain.invoke({"user_input":user_input})
    return result.strip().lower()
