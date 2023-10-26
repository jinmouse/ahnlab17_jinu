import time
from dotenv import load_dotenv

from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)


llm_model = "gpt-3.5-turbo"

load_dotenv()

def print_start() -> None:
  # 프로그램 시작 시간 기록
  global start_time
  start_time = time.time()
  print("프로그램 실행중...")

def print_end() -> None:
  # 프로그램 종료 시간 기록
  end_time = time.time()
  # 실행 시간 계산
  execution_time = end_time - start_time
  print(f"프로그램 종료: {execution_time} 초")

def main()->None:


  chat = ChatOpenAI(model=llm_model, temperature=0.9 )
  sys = SystemMessage(content="너는 대한민국 경제 전문가야. ")
  msg = HumanMessage(content='2010년 이후로 경제위기에 대해 분석해서, 앞으로 2023년 12월 주식 전망에 대해 설명해줘')

  print_start()
  aimsg = chat([sys, msg])
  print(aimsg.content)
  print_end()


if __name__ == '__main__':
  main()
