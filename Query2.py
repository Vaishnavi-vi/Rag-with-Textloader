from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace,HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import streamlit as st
import os

load_dotenv()

#document_loader
loader=TextLoader("cricket.txt",encoding="utf-8")

docs=loader.load()

#text splitter
splitter=RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=25
)
chnks=splitter.split_documents(docs)

#embedding
embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

#vector daabase
vector_store=Chroma.from_documents(
    embedding=embedding_model,
    documents=chnks,
    collection_name="my_collection"
)
#model
llm1=HuggingFaceEndpoint(repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",model_kwargs={"api_key":os.getenv("HUGGINGFACE_API_TOKEN")},temperature=0.7,max_new_tokens=512)
model=ChatHuggingFace(llm=llm1)
#retriever
#retrieve=vector_store.as_retriever(search_type="similarity",search_kwargs={"k":2})
#streamlit ui
st.set_page_config("Rag Question-Answer",layout="centered")
page=st.sidebar.radio("Select one between the two:",["Without LLM","With LLM"])
if page=="Without LLM":
    st.title("Rag Question-Answer without LLM")
    st.subheader("Please add your Question according to cricket.txt")
    query=st.text_input("Enter your Question:")
    if st.button("Answer:"):
        if query=="":
            st.warning("Please enter your Question")
        else:
            with st.spinner("Thinking..."):
                retrieve=vector_store.as_retriever(search_kwargs={"k":2},search_type="similarity")
                output=retrieve.invoke(query)
                st.success("Response")
                for i, doc in enumerate(output):
                    st.write(f"Result:{i}")
                    st.write(f"{output[i].page_content}")
                
elif page=="With LLM":
    st.title("Rag Question-Answer with llm")
    st.subheader("Please add your Question according to cricket.txt")
    query=st.text_input("Please enter your Question:")
    if st.button("Answer:"):
        if query=="":
            st.warning("Please enter your Question")
        else:
            with st.spinner("Thinking..."):
                retrieve=vector_store.as_retriever(search_kwargs={"k":2},search_type="similarity")
                retrieval_qa_=RetrievalQA.from_chain_type(
                    llm=model,
                    retriever=retrieve,
                    return_source_documents=True    
                )
                output=retrieval_qa_.invoke(query)
                st.success("Here is your answer")
                st.write(output["result"])
                
                
                
               
                
    
            