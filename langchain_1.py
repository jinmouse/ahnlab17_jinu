import os
import time
import json

from dotenv import load_dotenv


from langchain.chat_models import ChatOpenAI
from langchain.schema import (
  AIMessage,
  HumanMessage,
  SystemMessage
)

llm_model = "gpt-3.5-turbo"



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

load_dotenv()

def main()->None:
  chat = ChatOpenAI(model=llm_model)

  sys = SystemMessage(content="당신은 음악 추천을 해주는 전문 AI입니다.")
  msg = HumanMessage(content='1980년대 메탈 음악 5곡 추천해줘.')


  print_start()
  aimsg = chat([sys, msg])

  print(type(aimsg))
  print(aimsg.content)

  print_end()


if __name__ == '__main__':
  main()