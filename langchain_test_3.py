#!/usr/bin/env python
# coding: utf-8


import os
import time
import json
import sys
from typing import Any, Iterable, List
import langchain
from langchain.chat_models import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import PyPDFLoader
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

import openai

from dotenv import load_dotenv

load_dotenv()


openai.api_key = os.getenv("OPENAI_API_KEY")
openai.organization = os.getenv("ORGANIZATION")
sys.path.append(os.getenv("PYTHONPATH"))
llm_model = "gpt-3.5-turbo"
PDF_FREELANCER_GUIDELINES_FILE = "./data/프리랜서 가이드라인 (출판본).pdf"
CSV_OUTDOOR_CLOTHING_CATALOG_FILE = "data/OutdoorClothingCatalog_1000.csv"



# pdf 파일요약
# 질문 : map_reduce를 사용했는데도 시간이 많이 소요되는데 해결방법이 있을까요?
def summaryPDF():
  
  loader = PyPDFLoader(PDF_FREELANCER_GUIDELINES_FILE)
  docs = loader.load()

  llm_model = "gpt-3.5-turbo-1106"
  llm = ChatOpenAI(temperature=0.0, model=llm_model)
  
  chain = load_summarize_chain(llm, chain_type="map_reduce")

  summaryPdf =  chain.run(docs)
  
  prompt = ChatPromptTemplate.from_template(
    "'{summaryPdf}' 내용을 한글로 번역해줘"
  )

  chain = LLMChain(llm=llm, prompt=prompt)
  result = chain.run(summaryPdf)

  return result



if __name__ == '__main__':
  app.run(debug=False)
