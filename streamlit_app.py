import streamlit as st
from audio_recorder_streamlit import audio_recorder
import openai
import base64
import speech_recognition as sr
from dotenv import load_dotenv
import os
import wave
from gtts import gTTS
import pyaudio
import translators as ts
from langchain_community.document_loaders import TextLoader
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

class SpeechProcessor:
    def audio_to_text(self, audio_file, lang='en-US'):
        recognizer = sr.Recognizer()
        with sr.AudioFile(audio_file) as source:
            audio_data = recognizer.record(source)  # Record the audio file

        try:
            text = recognizer.recognize_google(audio_data, language=lang)
            return text
        except sr.UnknownValueError:
            print("Could not understand audio")
            return None
        except sr.RequestError as e:
            print(f"Error making the request; {e}")
            return None

    def text_to_audio(self, text, output_file="output.mp3", language='en'):
        tts = gTTS(text=text, lang=language)
        tts.save(output_file)
        # Play the audio (optional)
        # os.system("start " + output_file)

class LanguageTranslator:
    def translate(self, text, translator, from_language, to_language):
        translated_text = ts.translate_text(text, translator=translator, from_language=from_language, 
                                            to_language=to_language)
        return translated_text

class DocumentProcessor:
    def load_documents(self, file_path):
        loader = TextLoader(file_path)
        return loader.load()

    def process_documents(self, docs, api_key):
        embeddings = OpenAIEmbeddings(api_key=api_key) # default model=text-embedding-ada-002
        text_splitter = RecursiveCharacterTextSplitter()
        documents = text_splitter.split_documents(docs)
        vector = FAISS.from_documents(documents, embeddings) # returns a VectoreStore
        return vector

class ChatProcessor:
    def __init__(self, api_key):
        # Chat large language models API
        self.llm = ChatOpenAI(openai_api_key=api_key) # default model is gpt-3.5-turbo
        self.prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:
        <context>
        {context}
        </context>
        Question: {input}. Frame a sentence providing the building name, floor number and room number if applicable. If you are encountering questions that are not relevant to the conext, please respond that it is not a relevant question to the given context.""")

    def generate_response(self, text, vector):
        document_chain = create_stuff_documents_chain(self.llm, self.prompt)
        retriever = vector.as_retriever() # returns a VectorStoreRetriever
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        response = retrieval_chain.invoke({"input": text})
        return response["answer"]

# function to 
def main():
    # st.sidebar.title("API KEY CONFIGURATION")
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY") #st.sidebar.text_input("Enter you OpenAI API Key", type="password")
    st.title("Multilingual Voice Assistant🤖")
    st.header("For University Campus Navigation")
    option = st.selectbox('Language', ('English', 'Tamil', 'Hindi', 'Telugu'))
    
    if api_key:
        if option == 'Tamil':
            welcome_text = "வணக்கம்! வளாகத்தில் பயணம் செய்வது குறித்த உங்கள் கேள்விகளைக் கேட்க மைக்கைக் கிளிக் செய்யவும்."
            lang = 'ta-IN'
            question = "கேள்வி: "
            language_code = 'ta'
            answer = 'பதில்: '

        if option == 'English':
            welcome_text = "Hi there! Click on the voice recorder to ask your queries on navigating through the campus."
            lang = 'en-US'
            question = "Question: "
            language_code='en'
            answer = "Answer: "

        if option == 'Hindi':
            welcome_text = "नमस्ते! परिसर में भ्रमण के बारे में अपने प्रश्न पूछने के लिए वॉयस रिकॉर्डर पर क्लिक करें।"
            lang='hi-IN'
            question = 'सवाल: '
            language_code='hi'
            answer = "उत्तर: "
        
        if option == 'Telugu':
            welcome_text = 'హాయ్! క్యాంపస్‌లో నావిగేట్ చేయడంపై మీ ప్రశ్నలను అడగడానికి వాయిస్ రికార్డర్‌పై క్లిక్ చేయండి.'
            lang = 'te-IN'
            question = 'ప్రశ్న: '
            language_code = 'te'
            answer = "సమాధానం: "
        st.write(welcome_text)  
        recorded_audio = audio_recorder(#energy_threshold=(-1.0, 1.0),
            pause_threshold=2.0)
        
        if recorded_audio:
            audio_file_path = "C://Users//mercy.bai//projects//langchain//input_audio_data//recorded_audio.wav"
            with open(audio_file_path, "wb") as f:
                f.write(recorded_audio)

            audio_processor = SpeechProcessor()
            text_result = audio_processor.audio_to_text(audio_file_path, lang=lang)
            st.write(question, text_result)
            if text_result:
                # Translate the Question to English
                if option != 'English':
                    translator = LanguageTranslator()
                    text_result = translator.translate(text_result, translator='google', 
                                                       from_language=language_code, 
                                                       to_language="en")
                
                # Generate the response using Langchain
                doc_processor = DocumentProcessor()
                docs = doc_processor.load_documents("SRMISTLocationDetails.txt")
                vector = doc_processor.process_documents(docs, api_key)
                chat_processor = ChatProcessor(api_key)
                response = chat_processor.generate_response(text_result, vector)
                
                # Translate the response to the respective language from English
                if option != 'English':
                    response = translator.translate(response, translator="bing", from_language="en", 
                                                    to_language=language_code)
                    
                response_audio_file = "output_audio/answer_audio.mp3"
                audio_processor.text_to_audio(response, output_file=response_audio_file, language=language_code)
                st.audio(response_audio_file, format="audio/mp3")
                st.write(answer, response)
if __name__ == "__main__":
    main()