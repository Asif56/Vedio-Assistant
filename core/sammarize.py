from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough,RunnableLambda
# from langchain_groq import ChatGroq
from langchain_mistralai import ChatMistralAI

import os
from dotenv import load_dotenv
load_dotenv()

# GROQ_API_KEY=os.getenv('GROQ_API_KEY')
# os.environ['GROQ_API_KEY']=GROQ_API_KEY

def get_llm():
    return ChatMistralAI(
    model="mistral-small-latest",
    temperature=0.3,
    mistral_api_key = os.getenv("MISTRAL_API_KEY")
    )

def split_transcript(transcript:str) -> list:
    splitter= RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=200
    )
    return splitter.split_text(transcript)


def summarizes(transcript:str)-> str:
    llm = get_llm()
    map_prompt = ChatPromptTemplate.from_messages(
        [
            ("system","Summarize this portion of a meeting transcript concisely."),
            ('human',"{text}"),
        ]
    )
    map_chain = map_prompt | llm | StrOutputParser()
    chunks = split_transcript(transcript)
    chunks_summaries = [map_chain.invoke({"text": chunk}) for chunk in chunks]

    combined = "\n\n".join(chunks_summaries)

    combined_prmopt=ChatPromptTemplate.from_messages([
        (
            "system",
            "You are an expert meeting summarizer. Combine thses partial summaries "
            "into onre final professional meeting summary in bulle points",
        ),
        ("human","{text}")
    ]
    )
    combined_chain= (
        RunnablePassthrough() | RunnableLambda(lambda x:{"text":x}) | combined_prmopt | llm | StrOutputParser()
    )

    return combined_chain.invoke(combined)

def generate_title(transcript:str)->str:
    llm=get_llm()
    title_prompt = ChatPromptTemplate.from_messages(
        [
            (
             "system",
             'Based on the meeting transcript,generate a short professional meeting title'
             "(max 8 words). only return the title, nothing else."
            ),
            ("human","{text}"),
        ]
    )
    title_chain=(
        RunnablePassthrough() | RunnableLambda(lambda x:{"text":x}) | title_prompt | llm | StrOutputParser()
    )

    return title_chain.invoke(transcript[:2000])

