from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
import streamlit as st
import os
import time

from langchain.memory.buffer import ConversationBufferMemory
from langchain.prompts import PromptTemplate
import chainlit as cl
from langchain.chains import LLMChain

from chainlit import ChainlitApp
from chainlit.io.markdown import Markdown


from dotenv import load_dotenv

load_dotenv()


os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")

@cl.on_chat_start
def query_llm():
    llm=Ollama(model="llama2")
    
    conversation_memory = ConversationBufferMemory(memory_key="chat_history",
                                                   max_len=50,
                                                   return_messages=True,
                                                   )
    llm_chain = LLMChain(llm=llm, 
                         prompt=fitness_assistant_prompt_template,
                         memory=conversation_memory)
    
    cl.user_session.set("llm_chain", llm_chain)


@cl.on_message
async def query_llm(message: cl.Message):
    llm_chain = cl.user_session.get("llm_chain")
    
    response = await llm_chain.acall(message.content, 
                                     callbacks=[
                                         cl.AsyncLangchainCallbackHandler()])
    
    await cl.Message(response["text"]).send()



#Memory retrieval
conversation_memory = ConversationBufferMemory(memory_key="chat_history",
                                               max_len=50,
                                               return_messages=True
                                                   )




# streamlit framework

st.title('Fitness Chatbot With LLAMA2')
input_text=st.text_input("Search the topic u want related to fitness/health/gym")

# ollama LLAma2 LLm 
llm=Ollama(model="llama2")

# output_parser=StrOutputParser()
# chain=prompt|llm|output_parser


fitness_assistant_template = """
You are a fitness and gym assistant chatbot named "Fitsie". Your expertise is 
exclusively in providing information and advice about anything related to 
fitness and health. This includes diet, importance of workout, cardio, and general
fitness related queries. You do not provide information outside of this 
scope. If a question is not related to health,fitness, respond with, "I specialize 
only in fitness related queries."
Chat History: {chat_history}
Questions:{question}
Answer:"""

fitness_assistant_prompt_template = PromptTemplate(
    input_variables=["chat_history", "question"],
    template=fitness_assistant_template
)

llm_chain = LLMChain(llm=llm, prompt=fitness_assistant_prompt_template)






print("Loading chainlit module...")
from chainlit import ChainlitApp
print("ChainlitApp imported successfully!")
from chainlit.io import Markdown


md_file = "chainlit.md"
localama_modified = ChainlitApp(
    title="My Chainlit App",
    main_content=Markdown(md_file),
    # Add other settings as needed
)






