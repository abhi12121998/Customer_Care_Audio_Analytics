import os
import streamlit as st
from pytube import YouTube
import requests
import time
import random
import string
import pandas as pd
import csv
import json
import uvicorn
import shutil
from collections import defaultdict
from dotenv import load_dotenv
from langchain.chat_models import AzureChatOpenAI
from langchain.schema import HumanMessage
import sys


from utils import load_documents, \
                            create_vector_db, \
                            save_embedding_info, \
                            load_index
                            
load_dotenv()

os.environ["api_type"] = os.getenv('OPENAI_API_TYPE')
os.environ["api_version"] = os.getenv('OPENAI_API_VERSION')
os.environ["api_base"] = os.getenv('OPENAI_API_BASE')
os.environ["api_key"] = os.getenv("OPENAI_API_KEY")
os.environ["assembly_api"] = os.getenv("ASSEMBLYAI_AUTH_KEY")

# ====== This can be defined in the config file
user_input_chunk_size = 4000
user_input_chunk_overlap = 10
# =============================================

global chromma_db
global embedding_info_save_path
embedding_info_save_path = "./embedding_info/"
#global embedded_files_dir_path
global uploaded_files_dir_path
uploaded_files_dir_path = os.path.join("./embeding files/")
#embedded_files_dir_path = os.path.join(embedding_info_save_path, "embedded_files/")
os.makedirs(embedding_info_save_path, exist_ok = True)
os.makedirs(uploaded_files_dir_path, exist_ok = True)
#os.makedirs(embedded_files_dir_path, exist_ok = True)

UPLOAD_FOLDER = "./audio files"  # Replace with the actual path
ASSEMBLYAI_AUTH_KEY = os.environ["assembly_api"]  # Replace with your AssemblyAI API key

def simulate_long_running_process():
    time.sleep(10)  # Simulate a 1-second delay
def generate_random_filename(length=10, extension=".txt"):
    # Generate a random string of letters and digits
    random_string = ''.join(random.choices(string.ascii_letters + string.digits, k=length))

    # Combine with the file extension
    filename = random_string + extension

    return filename

def transcribe_audio(audio_file_path, categories=False):
    upload_endpoint = 'https://api.assemblyai.com/v2/upload'
    transcript_endpoint = "https://api.assemblyai.com/v2/transcript"

    headers_auth_only = {'authorization': ASSEMBLYAI_AUTH_KEY}
    headers = {
       "authorization": ASSEMBLYAI_AUTH_KEY,
       "content-type": "application/json"
    }
    CHUNK_SIZE = 5242880

    def read_file(filename):
        with open(filename, 'rb') as _file:
            while True:
                data = _file.read(CHUNK_SIZE)
                if not data:
                    break
                yield data

    # Upload the audio file to AssemblyAI
    upload_response = requests.post(
        upload_endpoint,
        headers=headers_auth_only,
        data=read_file(audio_file_path)
    )

    audio_url = upload_response.json()['upload_url']
    st.write('Uploaded to', audio_url)

    # Start the transcription of the audio file
    transcript_request = {
        'audio_url': audio_url,
		'iab_categories': 'True' if categories else 'False',
        "speaker_labels": True
    }

    transcript_response = requests.post(transcript_endpoint, json=transcript_request, headers=headers)

    # This is the id of the file that is being transcribed in the AssemblyAI servers
    # We will use this id to access the completed transcription
    transcript_id = transcript_response.json()['id']
    polling_endpoint = transcript_endpoint + "/" + transcript_id

    polling_response = requests.get(polling_endpoint, headers=headers)
    import time
    status = polling_response.json()['status']
    while status != 'completed':
        polling_response = requests.get(polling_endpoint, headers=headers)
        status = polling_response.json()['status']
        with st.spinner("Transcribing audio..."):
            simulate_long_running_process()

    random_filename = generate_random_filename()
    polling_response = requests.get(polling_endpoint, headers=headers)
    transcript = polling_response.json()['utterances']

    with open(f"./transcription/{random_filename}", 'w') as file:
        for i in range(len(transcript)):
            file.write(f"{transcript[i]['speaker']}==> {transcript[i]['text']}\n")

    os.remove(audio_file_path)

    return True


def Embeddings():
    try:

        document_chunks = create_vector_db(directory_path = uploaded_files_dir_path,
                                        user_input_chunk_size = user_input_chunk_size,
                                        user_input_chunk_overlap = user_input_chunk_overlap,
                                        embedding_info_save_path = embedding_info_save_path)
    except:
        st.write("Please check the audio files")

    return document_chunks

def qna():
    questions = ["Rate the agent's performance on a scale from 1 to 10 as if you were an employer. Provide only the numerical rating without text.",
                 "Please act as a employeer and Rate how much customer felt satisfied  from the conversation in a scale from 1 to 10 and Provide only the numerical rating without text",
                 "please tell how many times agent used abusive language, abbsuive terms used",
                 "please tell how many times customer used abusive language, abbsuive terms used",
                 "Tell me the intent of agent towards customer?and give answer only in one word out of this given choices ('positive', 'negative', 'neutral')",
                 "Tell me the intent of customer towards agent?and give answer only in one word ('positive', 'negative', 'neutral')",
                 "Did agent could able to solve the problem give answer in ('yes', 'No') only and don't give any additional text.",
                 "list down the information Which agent suggested to solve the problem?",
                 "List down the problems mentioned by the customer in the conversation?"
                 ]
    custom_qa_loaded, chroma_db = load_index(embedding_info_save_path)
    answers =[]
    for query in questions:
        response = custom_qa_loaded({"query": query})
        answers.append(response['result'])

    llm = AzureChatOpenAI(temperature = os.getenv("TEMPERATURE"), 
                        deployment_name = os.getenv("LLM_MODEL_DEPLOYMENT_NAME"),
                        model_kwargs = {
                                        "api_key": os.environ["OPENAI_API_KEY"],
                                        "api_base": os.environ["OPENAI_API_BASE"],
                                        "api_type": os.environ["OPENAI_API_TYPE"],
                                        "api_version": os.environ["OPENAI_API_VERSION"],
                                        })
    v = "give answer in numerical format from this list ['0,','1','2','3','4','5','6']"
    ans = llm([HumanMessage(content=f"Please tell how many times abbusive language used in given sentence==>: {answers[2]} {v}")])
    answers.append(ans.content)
    ans = llm([HumanMessage(content=f"Please tell how many times abbusive language used in given sentence==>: {answers[3]} {v}")])
    answers.append(ans.content)
    current_directory = os.getcwd()

# Define the file path for the CSV
    cost_analysis_record_save_path = os.path.join(current_directory, "data.csv")

    if not os.path.exists(cost_analysis_record_save_path):
        cost_analysis_record = pd.DataFrame(columns=["Agent Ratings",
                                                     "Customer satisifaction",
                                                     "Agent abbusived count",
                                                     "Agent abbusived",
                                                     "Customer abbusived count",
                                                     "Customer abbusived",
                                                     "Agent intent",
                                                     "Customer intent",
                                                     "Problem solved",
                                                     "Solutions given",
                                                     "probelms"])
        cost_analysis_record.to_csv(cost_analysis_record_save_path, index=False)

    
    new_data = [{"Agent Ratings":answers[0],
                "Customer satisifaction":answers[1],
                "Agent abbusived count":answers[9],
                "Agent abbusived":answers[2],
                "Customer abbusived count":answers[10],
                "Customer abbusived":answers[3],
                "Agent intent":answers[4],
                "Customer intent":answers[5],
                "Problem solved":answers[6],
                "Solutions given":answers[7],
                "probelms":answers[8],
                }]
    with open(cost_analysis_record_save_path, mode='a', newline='') as file:
        fieldnames = ["Agent Ratings",
                                                     "Customer satisifaction",
                                                     "Agent abbusived count",
                                                     "Agent abbusived",
                                                     "Customer abbusived count",
                                                     "Customer abbusived",
                                                     "Agent intent",
                                                     "Customer intent",
                                                     "Problem solved",
                                                     "Solutions given",
                                                     "probelms"]  # Define the column names/headers
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        # Write the new data rows
        for row in new_data:
            writer.writerow(row)
        
    return answers
            

def upload_audio_files_to_web_app():

    st.title("Customer Care Audio Analytics")
    upload_option = st.radio("Select upload option:", ["Local Path", "YouTube URL"])
    youtube_url = ""
    if upload_option=='Local Path':
        st.write("PLEASE SELECT THE AUDIO/VIDEO FILES FOR ANAYTICS")

        # Create a file uploader widget to select multiple audio files
        audio_files = st.file_uploader("Select multiple audio files", type=["mp3", "wav", "weba","ogg","mp4"], accept_multiple_files=True)

        if audio_files:
            st.write("Selected Audio Files:")
            for audio_file in audio_files:
                file_name = audio_file.name
                st.write(file_name)

                # Save the uploaded audio file to the specified folder
                destination_path = os.path.join(UPLOAD_FOLDER, file_name)
                with open(destination_path, "wb") as f:
                    f.write(audio_file.read())
                st.audio(destination_path)

                # Transcribe the uploaded audio file
                transcript = transcribe_audio(destination_path)
                st.success("Audio files have been successfully uploaded and transcribed.")

            
    elif upload_option == "YouTube URL":
            
        # Create an input field for entering the YouTube URL
            youtube_url = st.text_input("Enter a YouTube URL:")

     #       if st.button("Upload and Transcribe"):
            if youtube_url:
                try:
                    # Download the YouTube video as audio and save it
                    youtube = YouTube(youtube_url)
                    audio_stream = youtube.streams.filter(only_audio=True).first()
                    audio_stream.download(output_path=UPLOAD_FOLDER)
                    audio_file_path = os.path.join(UPLOAD_FOLDER, audio_stream.default_filename)

                    st.audio(audio_file_path)

                    # Transcribe the downloaded audio file
                    transcript = transcribe_audio(audio_file_path)
                    
                    st.success("YouTube audio has been successfully uploaded and transcribed.")
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
            else:
                st.warning("Please enter a valid YouTube URL.")
    
    return True
        

    

def main():

    upload_audio_files_to_web_app()
        
    all_files = os.listdir("./transcription/")
    files = [os.path.join("./transcription/", file) for file in all_files if os.path.isfile(os.path.join("./transcription/", file))]
    for k in files:
        shutil.move(k, uploaded_files_dir_path)

        Embeddings()
        answers = qna()
        shutil.rmtree("./embedding_info/saved_embeddings")
        shutil.rmtree("./embedding_info/embedding_doc_info")


        transcipt_files=  os.path.join("./embeding files/")

        # Use os.listdir to get a list of all files in the folder
        file_list = os.listdir(transcipt_files)

        # Iterate through the list of files and remove each one
        for file_name in file_list:
            file_path = os.path.join(transcipt_files, file_name)
            if os.path.isfile(file_path):
                os.remove(file_path)
    
        st.success("Analysis Done")
    
if __name__ == "__main__":
    main()
    
    
