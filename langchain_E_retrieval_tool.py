#!/usr/bin/env python
# coding: utf-8


import os
import time
import json
import sys
from typing import Any, Iterable, List
import langchain
from langchain.docstore.document import Document

import openai

from dotenv import load_dotenv

load_dotenv()


openai.api_key = os.getenv("OPENAI_API_KEY")
openai.organization = os.getenv("ORGANIZATION")
sys.path.append(os.getenv("PYTHONPATH"))
llm_model = "gpt-3.5-turbo"
PDF_FREELANCER_GUIDELINES_FILE = "./data/프리랜서 가이드라인 (출판본).pdf"
CSV_OUTDOOR_CLOTHING_CATALOG_FILE = "data/OutdoorClothingCatalog_1000.csv"

from langchain.vectorstores import FAISS
from langchain.vectorstores import Chroma
from langchain.schema.vectorstore import (
  VectorStore,
  VectorStoreRetriever
)
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from utils import (
  BusyIndicator,
  ConsoleInput,
  get_filename_without_extension,
  load_pdf_vectordb,
  load_vectordb_from_file,
  get_vectordb_path_by_file_path
  )
from langchain.chains.router import MultiRetrievalQAChain
from langchain.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.agents.agent_toolkits import create_conversational_retrieval_agent

import re

def reduce_newlines(input_string):
  # 정규 표현식을 사용하여 연속된 '\n'을 하나로 치환
  reduced_string = re.sub(r'\n{3,}', '\n\n', input_string)
  return reduced_string


def print_documents(docs: List[Any]) -> None:
  if docs == None:
    return

  print(f"documents size: {len(docs)}")
  p = lambda meta, key: print(f"{key}: {meta[key]}") if key in meta else None
  for doc in docs:
    print(f"source : {doc.metadata['source']}")
    p(doc.metadata, 'row')
    p(doc.metadata, 'page')
    print(f"content: {reduce_newlines(doc.page_content)[0:500]}")
    print('-'*30)

def print_result(result: Any) -> None:
  p = lambda key: print(f"{key}: {result[key]}") if key in result else None
  p('query')
  p('question')
  print(f"result: {'-' * 22}" )
  p('result')
  p('answer')
  print('-'*30)
  if 'source_documents' in result:
    print("documents")
    print_documents(result['source_documents'])


llm = ChatOpenAI(model_name=llm_model, temperature=0)



def get_personal_retriever() -> VectorStoreRetriever:
  personal_texts = [
    "내 이름은 홍길동입니다.",
    "내가 제일 좋아하는 색은 보라색입니다.",
    "내 꿈은 최고의 인공지능 활용 어플리케이션 개발자가 되는 것입니다.",
    "내 고향은 제주도입니다.",
    "나는 남성입니다",
    "나는 1972년에 태어났습니다.",
  ]
  personal_retriever = FAISS.from_texts(personal_texts, OpenAIEmbeddings()).as_retriever()
  if not isinstance(personal_retriever, VectorStoreRetriever):
    raise ValueError("personal_retriever is not VectorStoreRetriever")
  return personal_retriever


def get_freelancer_guidelines() -> VectorStoreRetriever:
  retriever = load_vectordb_from_file(PDF_FREELANCER_GUIDELINES_FILE).as_retriever()
  if not isinstance(retriever, VectorStoreRetriever):
    raise ValueError("it's not VectorStoreRetriever")
  return retriever

def get_outdoor_clothing_catalog() -> VectorStoreRetriever:
  retriever = load_vectordb_from_file(CSV_OUTDOOR_CLOTHING_CATALOG_FILE).as_retriever()
  if not isinstance(retriever, VectorStoreRetriever):
    raise ValueError("it's not VectorStoreRetriever")
  return retriever



def get_tools() :
  tools = [
    create_retriever_tool(
      get_freelancer_guidelines(),
      "freelancer_guidelines",
      "Good for answering questions about the different things you need to know about being a freelancer",
    ),
    create_retriever_tool(
      get_outdoor_clothing_catalog(),
      "outdoor_clothing_catalog",
      "Good for answering questions about outdoor clothing names and features",
    ),
    create_retriever_tool(
      get_personal_retriever(),
      "personal",
      "Good for answering questions about me",
    )
  ]
  return tools




def chat_qa(is_debug=False) -> None:
  console = ConsoleInput(basic_prompt='% ')
  busy_indicator = BusyIndicator().busy(True, "vectordb를 로딩중입니다 ")
  tools = get_tools()
  busy_indicator.stop()

  agent_executor = create_conversational_retrieval_agent(llm, tools, verbose=True)

  while True:  # 무한루프 시작
    t = console.input()[0].strip()

    if t == '':  # 빈 라인인 경우.
      continue

    if t == 'q' or t == 'Q' or t == 'ㅂ':
      break

    busy_indicator = BusyIndicator().busy(True)
    langchain.is_debug = is_debug
    result = agent_executor({"input": t})
    langchain.is_debug = False
    busy_indicator.stop()
    console.out(result["output"])
    if is_debug:
      print_result(result)






def input_select(menu: dict) -> (int, str):
  print(menu.get("title"))
  items = menu.get("items", None)

  if items == None or len(items) == 0:
     raise ValueError("menu에 items가 없습니다.")
  for idx, item in enumerate(items):
    print(f"{str(idx+1)}. {item}")

  size = len(items)
  select = -1

  while select < 0:
    try:
      select = int(input(">>선택 :"))
      if select <= 0 or size < select:
        select = -1
    except ValueError:
      select = -1

    if select < 0:
      print("잘못된 선택입니다.")

  return ( select, items[select-1] )




def main():
  debug, _ = input_select({
    "title" : "debugging 모드로 하시겠습니까?",
    "items" : [
      "yes",
      "no"
    ]
  })

  is_debug = debug == 1

  chat_qa( is_debug)



if __name__ == '__main__':
  main()
