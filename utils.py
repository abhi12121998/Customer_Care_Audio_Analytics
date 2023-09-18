


import os
import json
import pandas as pd
from collections import defaultdict
from dotenv import load_dotenv
from langchain.chat_models import AzureChatOpenAI
from langchain.chains import RetrievalQA
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.schema import Document
import chromadb
from llama_index import SimpleDirectoryReader




load_dotenv()

os.environ["api_type"] = os.getenv('OPENAI_API_TYPE')
os.environ["api_version"] = os.getenv('OPENAI_API_VERSION')
os.environ["api_base"] = os.getenv('OPENAI_API_BASE')
os.environ["api_key"] = os.getenv("OPENAI_API_KEY")



def load_documents(directory_path):
    documents = []
    try:
        documents_txt = DirectoryLoader(directory_path, glob="**/*.txt").load()
        documents.extend(documents_txt)
    except:
        print("Using SimpleDirectoryReader to read files.")
        documents_txt = SimpleDirectoryReader(directory_path, required_exts=".txt").load_data()
        for doc_num, doc_content in enumerate(documents_txt):
            metadata = {'source': str(SimpleDirectoryReader(directory_path, required_exts=".txt").input_files[doc_num])}
            document = Document(page_content = doc_content.text, metadata = metadata)
            documents.extend([document])
        print("number of txt documents ", len(documents_txt))

    return documents


#### creating vectordb and saving document information
def save_embedding_info(embeding_info, embedding_info_save_path):
    embedding_info_save_path = embedding_info_save_path + "/" if not embedding_info_save_path.endswith("/") else embedding_info_save_path

    document_info_save_path = embedding_info_save_path + "embedding_doc_info/"
    os.makedirs(document_info_save_path, exist_ok = True)

    embeding_info_dict = defaultdict(list)
    for doc_id, filename in zip(embeding_info["ids"], embeding_info["metadatas"]):
        # print(doc_id, "\n", os.path.basename(filename["source"]))
        embeding_info_dict[f"{os.path.basename(filename['source'])}"].append(doc_id)

    embeding_info_dict = dict(embeding_info_dict)
    document_info_save_file_path = document_info_save_path + "document_info.json"
    with open(document_info_save_file_path, "w") as json_file:
        json.dump(embeding_info_dict, json_file)

def create_vector_db(directory_path, user_input_chunk_size = 4000, 
                    user_input_chunk_overlap = 10, embedding_info_save_path = "./embedding_info/"):

    embedding_info_save_path = embedding_info_save_path + "/" if not embedding_info_save_path.endswith("/") else embedding_info_save_path
    os.makedirs(embedding_info_save_path, exist_ok = True)

    embeddings_save_path = embedding_info_save_path + "saved_embeddings/"
    os.makedirs(embeddings_save_path, exist_ok = True)

    # loading document
    documents = load_documents(directory_path)

    # splitting text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
                                chunk_size = user_input_chunk_size,
                                chunk_overlap = user_input_chunk_overlap,
                                length_function = len
                                )

    document_chunks = text_splitter.split_documents(documents)

    print("ingesting document to chromadb vector database")
    print(f"databse will be saved to local at: {embeddings_save_path}")
    # ingesting document_chunks into Chroma so we can efficiently query our embeddings:
    client_settings = chromadb.config.Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory=embeddings_save_path,
        anonymized_telemetry=False
    )
    #client_settings = chromadb.PersistentClient(path=embeddings_save_path)
    chroma_db_with_setting = Chroma(
        collection_name="langchain",
        embedding_function = OpenAIEmbeddings(deployment = os.getenv("EMBEDDING_MODEL_DEPLOYMENT_NAME"), model = os.getenv("EMBEDDING_MODEL_NAME"), chunk_size = 1, max_retries = 5),
        client_settings=client_settings,
        persist_directory=embeddings_save_path,
    )

    chroma_db_with_setting.add_documents(documents = document_chunks,
                            embedding=OpenAIEmbeddings(deployment = os.getenv("EMBEDDING_MODEL_DEPLOYMENT_NAME"), model = os.getenv("EMBEDDING_MODEL_NAME"), chunk_size = 1, max_retries = 5))

    embeding_info = chroma_db_with_setting.get()
    save_embedding_info(embeding_info, embedding_info_save_path)
    chroma_db_with_setting.persist()
    chroma_db_with_setting = None

    return document_chunks


### loading vector db and create qa chain for quering
def load_index(embedding_info_save_path):

    embedding_info_save_path = embedding_info_save_path + "/" if not embedding_info_save_path.endswith("/") else embedding_info_save_path
    embeddings_save_path = embedding_info_save_path + "saved_embeddings/"

    # creating prompt template
    custom_prompt_template = """I have agent and customer conversation transcript. as a quality perspective, i want to do the analysis of the given conversation {context} Question: {question} Answer only from the given document chunk in English:"""
    # Answer only from given documents:"""

    PROMPT = PromptTemplate(
                    template = custom_prompt_template, input_variables = ["context", "question"]
                    )
    chain_type_kwargs = {"prompt": PROMPT}

    llm = AzureChatOpenAI(temperature = os.getenv("TEMPERATURE"), 
                        deployment_name = os.getenv("LLM_MODEL_DEPLOYMENT_NAME"),
                        model_kwargs = {
                                        "api_key": os.environ["OPENAI_API_KEY"],
                                        "api_base": os.environ["OPENAI_API_BASE"],
                                        "api_type": os.environ["OPENAI_API_TYPE"],
                                        "api_version": os.environ["OPENAI_API_VERSION"],
                                        })

    client_settings = chromadb.config.Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory=embeddings_save_path,
        anonymized_telemetry=False
    )
    #client_settings = chromadb.PersistentClient(path=embeddings_save_path)

    chroma_db_for_qa_chain = Chroma(
        collection_name="langchain",
        embedding_function = OpenAIEmbeddings(deployment = os.getenv("EMBEDDING_MODEL_DEPLOYMENT_NAME"), model = os.getenv("EMBEDDING_MODEL_NAME"), chunk_size = 1, max_retries = 5),
        client_settings=client_settings,
        persist_directory=embeddings_save_path,
    )

    custom_qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", 
                                            retriever = chroma_db_for_qa_chain.as_retriever(type = "similarity"),
                                            chain_type_kwargs = chain_type_kwargs,
                                            return_source_documents = True
                                            )

    chroma_db_for_qa_chain.persist()
    return custom_qa, chroma_db_for_qa_chain


