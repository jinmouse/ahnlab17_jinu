


"""
파이썬으로 백엔드 서버 프로그램을 만드는 중.

각각 uri 별로 request 값과 result 값이 아래와 같은 서버 프로그램 코드를 작성하고  스웨거를 적용시켜줘.
flask-restx 를 사용 할 것

/new_token?db=<integer>

request : db : integer
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

import threading
from flask import Flask, request, send_from_directory
from flask_restx import Api, Resource, fields
import uuid
import time

import os
import json
import sys
# sys.path.append(os.getenv("PYTHONPATH"))

llm_model = "gpt-3.5-turbo"
PDF_FREELANCER_GUIDELINES_FILE = "./data/프리랜서 가이드라인 (출판본).pdf"
CSV_OUTDOOR_CLOTHING_CATALOG_FILE = "data/OutdoorClothingCatalog_1000.csv"
from utils import (
  BusyIndicator,
  ConsoleInput,
  get_filename_without_extension,
  load_pdf,
  load_pdf_vectordb,
  load_vectordb_from_file,
  get_vectordb_path_by_file_path
  )

# langchain 모듈
from langchain.chat_models import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import PyPDFLoader
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
persist_directory = 'docs/chroma/'
embedding = OpenAIEmbeddings()
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)



is_debug = True
app = Flask(__name__)


@app.route('/')
def serve_html():
  return send_from_directory('./html-docs', 'index.html')

@app.route('/<path:path>')
def serve_files(path):
  return send_from_directory('./html-docs', path)



api = Api(app, version='1.0', title='LangChain 기반 챗봇 API 서버', description='LangChain 기반 챗봇 API 서버로서 사용자의 입력에 따른 LLM 프롬프트 결과를 반환한다.', doc='/api-docs')

ns = api.namespace('api', description='API operations')



# Output Model for /new_token
token_output_model = api.model('TokenOutput', {
  'token': fields.String(description='Token string', required=True)  # 출력 모델
})

# Model for /prompt
prompt_model = api.model('PromptRequest', {
  'token': fields.String(description='Token string', required=True),
  'prompt': fields.String(description='Prompt string', required=True)
})

result_model = api.model('PromptResult', {
  'result': fields.String(description='Result string', required=True)
})


@ns.route('/new_token')
class NewTokenResource(Resource):
  @ns.doc(params={'db': 'A database identifier'})
  @ns.marshal_with(token_output_model, mask=False)  # 출력 모델 적용
  def get(self):
    db_value = request.args.get('db', type=int)  # URL query parameter에서 db 값을 가져옵니다.
    # 원하는 db 처리 로직을 여기에 추가하실 수 있습니다.
    return {'token': str(uuid.uuid4())}


request_idx = 0
@ns.route('/prompt')
class PromptResource(Resource):
  @ns.expect(prompt_model)
  @ns.marshal_with(result_model, mask=False)
  def post(self):
    data = request.json
    # You can process the prompt with the provided token here...
    # For the sake of this example, we just return the prompt string with "Processed:" prefix
    global request_idx
    idx = request_idx
    request_idx = request_idx + 1
    if is_debug:
      current_thread = threading.current_thread()
      print(f"{idx}.{data['token']} 현재 스레드: {current_thread.name} reqeust.")
    time.sleep(2)
    
    # print('summaryPDF ', '>'*80)
    # print(summaryPDF())
    
    
    if is_debug:
      print(f"{idx}.{data['token']} end.")
    return {'result': f'Processed: {data["prompt"]}'}


# pdf 파일요약
# 질문 : map_reduce를 사용했는데도 시간이 많이 소요되는데 해결방법이 있을까요?
def summaryPDF():
  
  loader = PyPDFLoader(PDF_FREELANCER_GUIDELINES_FILE)
  docs = loader.load()

  llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k")
  
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
