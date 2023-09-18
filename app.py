import os
import streamlit as st
import requests
import time
import random
import string
import pandas as pd
import csv
import shutil
from pytube import YouTube
from collections import defaultdict
from dotenv import load_dotenv
from langchain.chat_models import AzureChatOpenAI
from langchain.schema import HumanMessage

from download import download_button

from utils import load_documents, create_vector_db, load_index

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

global embedding_info_save_path
embedding_info_save_path = "./embedding_info/"
audio_files = "./audio files/"
trans = "./transcription/"
global uploaded_files_dir_path
uploaded_files_dir_path = os.path.join("./embeding files/")
os.makedirs(embedding_info_save_path, exist_ok=True)
os.makedirs(uploaded_files_dir_path, exist_ok=True)
os.makedirs(audio_files, exist_ok=True)
os.makedirs(trans, exist_ok=True)

UPLOAD_FOLDER = "./audio files"  # Replace with the actual path
ASSEMBLYAI_AUTH_KEY = os.environ["assembly_api"]  # Replace with your AssemblyAI API key

csv_file_path = "./data.csv"  # Replace with the actual path


# data = pd.read_csv(csv_file_path)
# download_button_str = download_button(data, '.\data.csv', 'Download Analysis data')
# st.markdown(download_button_str, unsafe_allow_html=True)

def simulate_long_running_process():
    time.sleep(10)  # Simulate a 1-second delay

def generate_random_filename(length=10, extension=".txt"):
    random_string = ''.join(random.choices(string.ascii_letters + string.digits, k=length))
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

    upload_response = requests.post(
        upload_endpoint,
        headers=headers_auth_only,
        data=read_file(audio_file_path)
    )

    audio_url = upload_response.json()['upload_url']
    st.write('Uploaded to', audio_url)

    transcript_request = {
        'audio_url': audio_url,
        'iab_categories': 'True' if categories else 'False',
        "speaker_labels": True
    }

    transcript_response = requests.post(transcript_endpoint, json=transcript_request, headers=headers)
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
        document_chunks = create_vector_db(directory_path=uploaded_files_dir_path,
                                           user_input_chunk_size=user_input_chunk_size,
                                           user_input_chunk_overlap=user_input_chunk_overlap,
                                           embedding_info_save_path=embedding_info_save_path)
    except:
        st.write("Please check the audio files")

    return document_chunks

def qna():
    questions = ["Rate the agent's performance on a scale from 1 to 10 as if you were an employer. Provide only the numerical rating without text.",
                 "Please act as an employer and rate how much the customer felt satisfied from the conversation on a scale from 1 to 10. Provide only the numerical rating without text.",
                 "Please tell how many times the agent used abusive language or abusive terms.",
                 "Please tell how many times the customer used abusive language or abusive terms.",
                 "Tell me the intent of the agent towards the customer. Provide a one-word answer from these choices: 'positive', 'negative', 'neutral'.",
                 "Tell me the intent of the customer towards the agent. Provide a one-word answer from these choices: 'positive', 'negative', 'neutral'.",
                 "Did the agent solve the problem? Provide an answer in 'yes' or 'no' without additional text.",
                 "List down the information which the agent suggested to solve the problem.",
                 "List down the problems mentioned by the customer in the conversation."
                 ]
    custom_qa_loaded, chroma_db = load_index(embedding_info_save_path)
    answers = []
    for query in questions:
        response = custom_qa_loaded({"query": query})
        answers.append(response['result'])

    llm = AzureChatOpenAI(temperature=os.getenv("TEMPERATURE"),
                          deployment_name=os.getenv("LLM_MODEL_DEPLOYMENT_NAME"),
                          model_kwargs={
                              "api_key": os.environ["OPENAI_API_KEY"],
                              "api_base": os.environ["OPENAI_API_BASE"],
                              "api_type": os.environ["OPENAI_API_TYPE"],
                              "api_version": os.environ["OPENAI_API_VERSION"],
                          })
    v = "Provide an answer in numerical format from this list ['0', '1', '2', '3', '4', '5', '6']"
    ans = llm([HumanMessage(content=f"Please tell how many times abusive language was used in the given sentence: {answers[2]}. {v}")])
    answers.append(ans.content)
    ans = llm([HumanMessage(content=f"Please tell how many times abusive language was used in the given sentence: {answers[3]}. {v}")])
    answers.append(ans.content)
    current_directory = os.getcwd()

    # Define the file path for the CSV
    cost_analysis_record_save_path = os.path.join(current_directory, "data.csv")

    if not os.path.exists(cost_analysis_record_save_path):
        cost_analysis_record = pd.DataFrame(columns=["Agent Ratings",
                                                     "Customer satisfaction",
                                                     "Agent abusive count",
                                                     "Agent abusive",
                                                     "Customer abusive count",
                                                     "Customer abusive",
                                                     "Agent intent",
                                                     "Customer intent",
                                                     "Problem solved",
                                                     "Solutions given",
                                                     "Problems"])
        cost_analysis_record.to_csv(cost_analysis_record_save_path, index=False)

    new_data = [{"Agent Ratings": answers[0],
                 "Customer satisfaction": answers[1],
                 "Agent abusive count": answers[9],
                 "Agent abusive": answers[2],
                 "Customer abusive count": answers[10],
                 "Customer abusive": answers[3],
                 "Agent intent": answers[4],
                 "Customer intent": answers[5],
                 "Problem solved": answers[6],
                 "Solutions given": answers[7],
                 "Problems": answers[8],
                 }]
    with open(cost_analysis_record_save_path, mode='a', newline='') as file:
        fieldnames = ["Agent Ratings",
                      "Customer satisfaction",
                      "Agent abusive count",
                      "Agent abusive",
                      "Customer abusive count",
                      "Customer abusive",
                      "Agent intent",
                      "Customer intent",
                      "Problem solved",
                      "Solutions given",
                      "Problems"]  # Define the column names/headers
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        # Write the new data rows
        for row in new_data:
            writer.writerow(row)

    return answers

def upload_audio_files_to_web_app():
    st.title("Customer Care Audio Analytics")
    upload_option = st.radio("Select upload option:", ["Local Path", "YouTube URL"])
    youtube_url = ""
    if upload_option == 'Local Path':
        st.write("PLEASE SELECT THE AUDIO/VIDEO FILES FOR ANALYTICS")

        # Create a file uploader widget to select multiple audio files
        audio_files = st.file_uploader("Select multiple audio files", type=["mp3", "wav", "weba", "ogg", "mp4"],
                                       accept_multiple_files=True)

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

        transcipt_files = os.path.join("./embeding files/")

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
    data = pd.read_csv(csv_file_path)
    download_button_str = download_button(data, '.\data.csv', 'Download Analysis data')
    st.markdown(download_button_str, unsafe_allow_html=True)
