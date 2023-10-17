import os
import time
import json

import openai

from dotenv import load_dotenv

load_dotenv()


openai.api_key = os.getenv("OPENAI_API_KEY")
openai.organization = os.getenv("ORGANIZATION")


saved_messages = [
  {
    "role": "system",
    "content": "너는 ChatGPT 도우미다"
  }
]

def query_prompt(messages: []):
  response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=messages,
    stream=True
  )

  return response

# 초록색 ansi code 값 반환
def color_code(is_green: bool) -> str:
  if is_green:
    return "\033[32m"  # 초록색으로 설정
  else:
    return "\033[0m"   # 색상 설정 초기화


def get_inputs() -> [str]:
  lines = []  # 사용자로부터 입력받은 문장을 저장할 리스트
  blank_count = 0  # 연속으로 입력받은 빈 줄의 수를 저장할 변수

  while True:
    user_input = input('$ ')  # 사용자로부터 문장 입력 받기

    if not user_input.strip():  # 입력받은 문장이 빈 줄이면
      blank_count += 1  # 빈 줄 카운트 증가

      if blank_count < 2:  # 빈 줄이 하나만 있을 경우에만 리스트에 추가
        lines.append('')
    else:
      blank_count = 0  # 빈 줄이 아니면 카운트 리셋
      lines.append(user_input)  # 문장 저장

    if blank_count == 2:  # 빈 줄을 두 번 연속으로 입력받으면
      break  # 내부 루프 종료
  return lines


def chat() -> None:
  while True:  # 무한루프 시작
    lines = get_inputs()

    t = ''.join(lines).strip()
    if t == '':  # lines 리스트에 모두 빈 라인인 경우.
      continue

    if t == 'q' or t == 'Q':
      break

    saved_messages.append({
      "role": "user",
      "content":  "\n".join(lines)
    })
    response = query_prompt(saved_messages)

    collected_chunks = []
    collected_messages = []
    print(color_code(True), end='')
    # iterate through the stream of events
    for chunk in response:
        collected_chunks.append(chunk)  # save the event response
        chunk_message = chunk['choices'][0]['delta']  # extract the message
        collected_messages.append(chunk_message)  # save the message
        print(chunk_message.get('content', ''), end='')  # print the delay and text
    print(color_code(False))

    saved_messages.append({
      "role": "assistant",
      "content":  ''.join([m.get('content', '') for m in collected_messages])
    })




if __name__ == '__main__':
  print("chatGPT를 실행합니다.")
  chat()
  print("프로그램을 종료합니다.")



