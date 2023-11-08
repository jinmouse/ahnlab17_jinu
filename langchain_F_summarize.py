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
  load_pdf,
  load_pdf_vectordb,
  load_vectordb_from_file,
  get_vectordb_path_by_file_path
  )
from langchain.chains.router import MultiRetrievalQAChain
from langchain.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.agents.agent_toolkits import create_conversational_retrieval_agent


from langchain.chains.summarize import load_summarize_chain

def main():
  docs = load_pdf(PDF_FREELANCER_GUIDELINES_FILE)

  llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k")
  chain = load_summarize_chain(llm, chain_type="map_reduce")

  result =  chain.run(docs)
  print(result)


"""
The passage discusses a book called "Freelancer Guidelines" by KarenJ, which provides valuable information and advice for freelancers in the software development industry. It covers topics such as starting as a freelancer, tax knowledge, interviews and contracts, managing downtime and vacations, negotiation skills, necessary equipment, building relationships, identifying good and bad companies, planning for the future, and calculating freelance income. The passage also mentions another book by Karen J called "Good Company / Bad Company" and provides information on tax knowledge for freelancers, including reporting and paying income tax, bookkeeping methods, tax deductions, income calculation methods, progressive taxation, and case examples. It briefly mentions font ownership and the importance of continuous tax management.

이 글에서는 소프트웨어 개발 업계의 프리랜서에게 유용한 정보와 조언을 제공하는 KarenJ의 "프리랜서 가이드라인"이라는 책에 대해 설명합니다. 이 책에서는 프리랜서로 시작하기, 세금 지식, 인터뷰 및 계약, 다운타임 및 휴가 관리, 협상 기술, 필요한 장비, 관계 구축, 좋은 회사와 나쁜 회사 식별, 미래 계획, 프리랜서 수입 계산과 같은 주제를 다룹니다. 이 구절은 또한 "좋은 회사/나쁜 회사"라는 Karen J의 또 다른 책을 언급하며 소득세 신고 및 납부, 기장 방법, 세금 공제, 소득 계산 방법, 누진세, 사례 등 프리랜서를 위한 세금 지식에 대한 정보를 제공합니다. 폰트 소유권과 지속적인 세무 관리의 중요성에 대해서도 간략하게 언급하고 있습니다.

"""


if __name__ == '__main__':
  main()
