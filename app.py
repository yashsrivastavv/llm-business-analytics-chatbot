import streamlit as st
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
# from datasets import load_dataset
import os

load_dotenv()
loader = CSVLoader(file_path="sales_conversations.csv")
docs = loader.load()


embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(docs, embeddings)

# print(len(docs))

#information retrieval

def retrieve_info(query):
    similar_response = db.similarity_search(query, k=3)

    page_contents_array = [doc.page_content for doc in similar_response]

    return page_contents_array


cust_msg = """Hello bot, I need to buy a new laptop."""
response = retrieve_info(cust_msg)
print(response)



#setting up LLM model
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")
template = """
You are a world class business development representative. 
I will share a prospect's message with you and you will give me the best answer that 
I should send to this prospect based on past best practies, 
and you will follow ALL of the rules below:

1/ Response should be very similar or even identical to the past best practies, 
in terms of length, tone of voice, logical arguments and other details

2/ If the best practice are irrelevant, then try to mimic the style of the best practice to prospect's message

Below is a message I received from the prospect:
{message}

Here is a list of best practies of how we normally respond to prospect in similar scenarios:
{best_practice}

Please write the best response that I should send to this prospect:
"""

prompt = PromptTemplate(
    input_variables=["message", "best_practice"],
    template=template
)

chain = LLMChain(llm=llm, prompt=prompt)

#generating responses

def generate_response(msg):
    bes_practice = retrieve_info(msg)
    response = chain.run(msg=msg, bes_practice=bes_practice)
    return response


#using strealit for the app

def main():
    st.set_page.config(
        page_title = "Response Generator", page_icon = ":bird:"
    )

    st.header("Response Generator...")
    msg=st.text_area("cutomer message")

    if msg:
        st.write("Generating best practice messages...")
        result = generate_response(msg)
        st.info(result)

if __name__== '__main__':
    main()

# print(docs[0])