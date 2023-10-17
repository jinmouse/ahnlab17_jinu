import os
import time
import json

import openai

from dotenv import load_dotenv

load_dotenv()


openai.api_key = os.getenv("OPENAI_API_KEY")
openai.organization = os.getenv("ORGANIZATION")

def query_prompt():
  response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
      {
        "role": "system",
        "content": "너는 한국어와 영어, 일본어를 잘 사용할 줄 아는 사람이다. 대답은 영어와 일어로만 말해야 한다."
      },
      {
        "role": "user",
        "content": "오늘 하루는 어때?"
      }
    ],
    temperature=0.53,
    max_tokens=256,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
  )

  return response


if __name__ == '__main__':
  # 프로그램 시작 시간 기록
  start_time = time.time()
  print("프로그램 실행중...")

  response = query_prompt()
  print(response)
  print('============')
  print(response.choices[0].message["content"])

  # 프로그램 종료 시간 기록
  end_time = time.time()
  # 실행 시간 계산
  execution_time = end_time - start_time
  print(f"프로그램 종료: {execution_time} 초")
