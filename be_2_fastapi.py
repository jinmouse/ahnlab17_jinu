


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
from langchain.chains import RetrievalQA
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

llm_model = "gpt-3.5-turbo"
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
  db: str


class PromptResult(BaseModel):
  result: str


# @app.get("/")
# async def serve_html():
#   return FileResponse('./html-docs/index.html')


@app.get("/api/new_token")
async def new_token(db: int):

  # 원하는 db 처리 로직을 여기에 추가하실 수 있습니다.
  return jsonable_encoder(TokenOutput(token=str(uuid.uuid4()), db=str(db)))


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
  
  # 여기에 langchain 적용  
  await asyncio.sleep(5)  # 예시를 위한 비동기 작업 (1초 대기)
  
  ################## start
  db = request.db
  global promptResult
  questionPrompt = request.prompt

  vectordb : FAISS = load_vectordb_from_file(PDF_FREELANCER_GUIDELINES_FILE if db == 1 else CSV_OUTDOOR_CLOTHING_CATALOG_FILE)
  
  if db == 1: # PDF 관련
    inclusion_result = check_inclusion(questionPrompt, "요약")
    if inclusion_result:
      # 전체 글에 대한 요약본을 볼 수 있도록 하라.
      chainType =  'map_reduce'
      promptResult = getAnswerPDF(vectordb, questionPrompt, chainType)
    else:
      # chainType =  'stuff'
      promptResult = test_pdf_splitter_embedding_vectorstore(questionPrompt)
      
  else:   # CSV 관련
    
    inclusion_result = len(questionPrompt)
    if inclusion_result == 0:
      # busy_indicator = BusyIndicator().busy(True, "vectordb를 로딩중입니다 ")
      tools = get_tools()
      agent_executor = create_conversational_retrieval_agent(llm, tools, verbose=True)
      result = agent_executor({"input": "상품 정보에 대하여 한국어로 번역해서 간략하게 안내를 해줘. 인사도 포함해서 친절하게 안내를 부탁해"})
      promptResult = result["output"]
      # busy_indicator.stop()
         
    else:
      df: pd.DataFrame = pd.read_csv(CSV_OUTDOOR_CLOTHING_CATALOG_FILE)
      print(df.head())
      promptResult = getLLMChain(df)
  

 ######################## end
  
  if is_debug:
    print(f"{idx}.{request.token} end.")
    
  return jsonable_encoder(PromptResult(result=f"Processed: {promptResult}"))


# >>>>>>>> def start

def getLLMChain(df: pd.DataFrame) -> None:
  llm = ChatOpenAI(temperature=0.9, model=llm_model)

 
  prompt = ChatPromptTemplate.from_template(
    """{product}상품에 대한 전반적인 통계 정보를 안내하라. \
      예) 총 상품수, 성별 의류수, 계절 별 의류수, 의류종류별 의류수 등
    """
  )
  product = df.head()
  chain = LLMChain(llm=llm, prompt=prompt)

  # print(chain.run(product))
  return chain.run(product)



# def get_freelancer_guidelines() -> VectorStoreRetriever:
#   retriever = load_vectordb_from_file(PDF_FREELANCER_GUIDELINES_FILE).as_retriever()
#   if not isinstance(retriever, VectorStoreRetriever):
#     raise ValueError("it's not VectorStoreRetriever")
#   return retriever

def get_outdoor_clothing_catalog() -> VectorStoreRetriever:
  retriever = load_vectordb_from_file(CSV_OUTDOOR_CLOTHING_CATALOG_FILE).as_retriever()
  if not isinstance(retriever, VectorStoreRetriever):
    raise ValueError("it's not VectorStoreRetriever")
  return retriever



def get_tools() :
  tools = [
    # create_retriever_tool(
    #   get_freelancer_guidelines(),
    #   "freelancer_guidelines",
    #   "Good for answering questions about the different things you need to know about being a freelancer",
    # ),
    create_retriever_tool(
      get_outdoor_clothing_catalog(),
      "outdoor_clothing_catalog",
      "Good for answering questions about outdoor clothing names and features",
    )
  ]
  return tools


# 전체 글에 대한 요약본 내용 조회
def getAnswerPDF( vectordb, questionPrompt, chainType)-> str:

  qa_chain_mr = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever(),
    chain_type=chainType
  )
  result = qa_chain_mr({"query": questionPrompt})

  return result["result"]



# 사용자 별로 히스토리를 관리하라. 히스토리는 총 5개 질문까지만 저장해도 된다.
def test_ConversationBufferWindowMemory(questionPrompt) -> str:

  memory = ConversationBufferWindowMemory(k=5)
  conversation = ConversationChain(
    llm=llm,
    memory = memory,
    verbose=False
  )
  result = conversation.predict(input=questionPrompt)
  
  print('memory.load_memory_variables ','>'*80)
  print(memory.load_memory_variables({}))

  return result


# 특정 단어 포함 여부 체크
def check_inclusion(target_string, substring="요약"):
    return substring in target_string


# TEST용 
# >> 
def test_pdf_splitter_embedding_vectorstore(questionPrompt):
  
  loader = PyPDFLoader(PDF_FREELANCER_GUIDELINES_FILE)
  
  # split documents
  text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=150)

  index = VectorstoreIndexCreator(
        vectorstore_cls=FAISS,
        embedding=OpenAIEmbeddings(),
        text_splitter= text_splitter
  ).from_loaders([loader])
  
  # index.vectorstore.save_local('asdf')
  
  result = index.query(questionPrompt)
  return result

# >> pdf 임베딩하고 벡터스토어에 저장
def test_pdf_vectordb(questionPrompt) -> str:
  vectordb : VectorStore = load_pdf_vectordb(PDF_FREELANCER_GUIDELINES_FILE)
  docs = vectordb.similarity_search(questionPrompt,k=1)

  # print(f"len(docs)=>{len(docs)}")
  # print(f"docs[0].page_content=>{docs[0].page_content}")
  
  result = docs[0].page_content
  return result

# >> 
def test_prompt(vectordb, question)-> None:
  
  #  
  template = """답을 모르는 경우, 답을 지어내려고 하지 말고 모른다고만 말하세요. \
    가능한 한 간결하게 답변하세요. 
    {context}
  """
  QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
  
  # Run chain
  qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever(),
    return_source_documents=False,
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
  )

  result = qa_chain({"query": question})

  return result

# >>>>>>>> def end


app.mount("/", StaticFiles(directory="./html-docs", html=True), name="static")


if __name__ == "__main__":
  import uvicorn
  uvicorn.run(app, host="0.0.0.0", port=5000)
