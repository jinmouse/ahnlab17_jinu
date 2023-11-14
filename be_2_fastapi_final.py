


"""
파이썬으로 백엔드 서버 프로그램을 만드는 중.

각각 uri 별로 request 값과 result 값이 아래와 같은 서버 프로그램 코드를 작성하고  스웨거를 적용시켜줘.
flask-restx 를 사용 할 것

/new_token

request : {
  db : integer
}
result : {
  token: string
}


/prompt

request : {
  token: string
  prompt: string
}

result : {
  result: string
}

"""
import os
import json
import sys
sys.path.append(os.getenv("PYTHONPATH", "./libs"))

from utils import (
  BusyIndicator,
  ConsoleInput,
  get_filename_without_extension,
  load_pdf,
  load_pdf_vectordb,
  load_vectordb_from_file,
  get_vectordb_path_by_file_path
  )


import threading
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
import uuid
import asyncio

# langchain 모듈
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.chains import ConversationChain
from langchain.chains import RetrievalQA,  ConversationalRetrievalChain
from langchain.chains.summarize import load_summarize_chain
from langchain.memory import ConversationBufferWindowMemory
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import PyPDFLoader
from langchain.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate
from langchain.schema.vectorstore import (
  VectorStore,
  VectorStoreRetriever
)
from langchain.vectorstores import FAISS
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.agents.agent_toolkits import create_conversational_retrieval_agent

import pandas as pd

import openai

from dotenv import load_dotenv

load_dotenv()


openai.api_key = os.getenv("OPENAI_API_KEY")
openai.organization = os.getenv("ORGANIZATION")

llm_model = "gpt-3.5-turbo-1106"
llm = ChatOpenAI(temperature=0.0, model=llm_model)

PDF_FREELANCER_GUIDELINES_FILE = "./data/프리랜서 가이드라인 (출판본).pdf"
CSV_OUTDOOR_CLOTHING_CATALOG_FILE = "data/OutdoorClothingCatalog_1000.csv"

is_debug = True
app = FastAPI(debug=is_debug, docs_url="/api-docs")


class TokenOutput(BaseModel):
  token: str
  db : str


class PromptRequest(BaseModel):
  token: str
  prompt: str
  db : str


class PromptResult(BaseModel):
  result: str
  token: str


# @app.get("/")
# async def serve_html():
#   return FileResponse('./html-docs/index.html')


vectordb: FAISS  # 전역 변수 선언
# vectordb: VectorStore  # 전역 변수 선언

@app.get("/api/new_token")
async def new_token(db: str):
  # 원하는 db 처리 로직을 여기에 추가하실 수 있습니다.
  # print('----->', db)
  global vectordb
  
  if db == "1":
    vectordb  = load_vectordb_from_file(PDF_FREELANCER_GUIDELINES_FILE)
  else:
    vectordb  = load_vectordb_from_file(CSV_OUTDOOR_CLOTHING_CATALOG_FILE)
    
  return jsonable_encoder(TokenOutput(token=str(uuid.uuid4()), db=db))

request_idx = 0

@app.post("/api/prompt")
async def process_prompt(request: PromptRequest):
  # 비동기적으로 처리할 내용을 여기에 구현합니다.
  # 예를 들어, 외부 API 호출이나 무거운 계산 작업 등을 비동기로 수행할 수 있습니다.
  global request_idx
  idx = request_idx
  request_idx = request_idx + 1
  if is_debug:
    current_thread = threading.current_thread()
    print(f"{idx}.{request.token} 현재 스레드: {current_thread.name} reqeust.")
    print(f"{idx}.{request.token} reqeust.")
  await asyncio.sleep(5)  # 예시를 위한 비동기 작업 (1초 대기)
  
  # start
  token = request.token
  db = request.db
  questionPrompt = request.prompt
  
  if db == '1': # PDF
    qa = get_qa(vectordb)
    
    result = qa({"question": questionPrompt})
    resultPrompt = result['answer']
    
  else:  # CVS
    if len(questionPrompt) == 0:
      questionPrompt = "전체 상품 정보에 대하여 간략하게 안내를 해줘. 인사도 포함해서 친절하게 안내를 부탁해. 답은 꼭 한국어로 번역해서 해줘. "
      
    tools = get_tools()
    agent_executor = create_conversational_retrieval_agent(llm, tools, verbose=True)
    result = agent_executor({"input": questionPrompt})
    resultPrompt = result["output"]
  # end
  
  
  if is_debug:
    print(f"{idx}.{request.token} end.")
  return jsonable_encoder(PromptResult(result=f"Processed: {resultPrompt}", token=token))


def get_qa(vectordb) -> ConversationalRetrievalChain:
  memory = ConversationBufferMemory(
    memory_key="chat_history",
    output_key='answer',
    return_messages=True
  )
  retriever=vectordb.as_retriever()
  qa = ConversationalRetrievalChain.from_llm(
      llm,
      retriever=retriever,
      return_source_documents=True,
      return_generated_question=True,
      memory=memory
  )
  return qa

def get_tools() :
  tools = [
    create_retriever_tool(
      get_outdoor_clothing_catalog(),
      "outdoor_clothing_catalog",
      "Good for answering questions about outdoor clothing names and features",
    )
  ]
  return tools

def get_outdoor_clothing_catalog() -> VectorStoreRetriever:
  retriever = load_vectordb_from_file(CSV_OUTDOOR_CLOTHING_CATALOG_FILE).as_retriever()
  if not isinstance(retriever, VectorStoreRetriever):
    raise ValueError("it's not VectorStoreRetriever")
  return retriever



app.mount("/", StaticFiles(directory="./html-docs", html=True), name="static")


if __name__ == "__main__":
  import uvicorn
  uvicorn.run(app, host="0.0.0.0", port=5000)
