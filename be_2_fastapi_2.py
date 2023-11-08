


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
import threading
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
import uuid
import asyncio

import os
import sys
import openai
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()


openai.api_key = os.getenv("OPENAI_API_KEY")
openai.organization = os.getenv("ORGANIZATION")
sys.path.append(os.getenv("PYTHONPATH"))
llm_model = "gpt-3.5-turbo"

llm = ChatOpenAI(model_name=llm_model, temperature=0)


is_debug = True
app = FastAPI(debug=is_debug, docs_url="/api-docs")


class TokenOutput(BaseModel):
  token: str


class PromptRequest(BaseModel):
  token: str
  prompt: str


class PromptResult(BaseModel):
  result: str


# @app.get("/")
# async def serve_html():
#   return FileResponse('./html-docs/index.html')



tokens = {

}
@app.get("/api/new_token")
async def new_token(db: int):
  # 원하는 db 처리 로직을 여기에 추가하실 수 있습니다.
  token=str(uuid.uuid4())
  tools = get_tools()
  agent_executor = create_conversational_retrieval_agent(llm, tools, verbose=True)
  tokens[token] = agent_executor
  return jsonable_encoder(TokenOutput(token=token))

request_idx = 0

from langchain_E_retrieval_tool import get_tools
from langchain.agents.agent_toolkits import create_conversational_retrieval_agent


# tools = get_tools()
# agent_executor = create_conversational_retrieval_agent(llm, tools, verbose=True)

@app.post("/api/prompt")
async def process_prompt(request: PromptRequest):
  # 비동기적으로 처리할 내용을 여기에 구현합니다.
  # 예를 들어, 외부 API 호출이나 무거운 계산 작업 등을 비동기로 수행할 수 있습니다.
  global request_idx
  idx = request_idx
  request_idx = request_idx + 1

  executor = tokens[request.token]
  if not executor:
    raise ValueError("token이 없습니다.")
  result = executor({"input": request.prompt})

  return jsonable_encoder(PromptResult(result=result["output"]))


app.mount("/", StaticFiles(directory="./html-docs", html=True), name="static")


if __name__ == "__main__":
  import uvicorn
  uvicorn.run(app, host="0.0.0.0", port=5000)
