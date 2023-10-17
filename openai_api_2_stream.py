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
        "content": "너는 ChatGPT 도우미다"
      },
      {
        "role": "user",
        "content": "API 사용료를 알려줘"
      }
    ],
    temperature=0.53,
    max_tokens=256,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0,
    stream=True
  )

  return response


if __name__ == '__main__':
  # 프로그램 시작 시간 기록
  start_time = time.time()
  print("프로그램 실행중...")

  response = query_prompt()

  # create variables to collect the stream of chunks
  collected_chunks = []
  collected_messages = []
  # iterate through the stream of events
  for chunk in response:
      chunk_time = time.time() - start_time  # calculate the time delay of the chunk
      collected_chunks.append(chunk)  # save the event response
      chunk_message = chunk['choices'][0]['delta']  # extract the message
      collected_messages.append(chunk_message)  # save the message
      # print(f"Message received {chunk_time:.2f} seconds after request: {chunk_message.get('content', '')}")  # print the delay and text
      print(chunk_message.get('content', ''), end='')

  # print the time delay and text received
  print(f"Full response received {chunk_time:.2f} seconds after request")
  full_reply_content = ''.join([m.get('content', '') for m in collected_messages])
  print(f"Full conversation received: {full_reply_content}")

  # 프로그램 종료 시간 기록
  end_time = time.time()
  # 실행 시간 계산
  execution_time = end_time - start_time
  print(f"프로그램 종료: {execution_time} 초")



