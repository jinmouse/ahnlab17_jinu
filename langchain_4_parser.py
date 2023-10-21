import os
import time
import json
import asyncio
from typing import List

from dotenv import load_dotenv


from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts.chat import (
  ChatPromptTemplate,
  SystemMessagePromptTemplate,
  HumanMessagePromptTemplate,
)
from langchain.callbacks.base import BaseCallbackHandler
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser


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


llm_model = "gpt-3.5-turbo"

load_dotenv()



class MyCustomHandler(BaseCallbackHandler):
  def on_llm_new_token(self, token: str, **kwargs) -> None:
    print(token, end='')


def get_response_schemas() -> List[ResponseSchema]:
  gift_schema = ResponseSchema(name="gift",
                             description="다른 사람을 위한 선물로 구매했는가? \
                              예인 경우 True, \
                              그렇지 아닌경우는 False 값을 설정한다.")
  delivery_days_schema = ResponseSchema(name="delivery_days",
                                      description="제품이 도착하는 데 도착하는 데 며칠이 걸렸나? \
                                        이 정보를 찾을 수 없다면 -1 값을 설정한다.")
  price_value_schema = ResponseSchema(name="price_value",
                                    description="값이나 가격에 대한 문장을 추출하여 쉼표로 구분된 Python 목록으로 출력한다.")

  return [gift_schema,
    delivery_days_schema,
    price_value_schema]


def get_parser() -> StructuredOutputParser:
  return StructuredOutputParser.from_response_schemas(get_response_schemas())




def main() -> None:
  handler = MyCustomHandler()
  chat = ChatOpenAI(temperature=0, model=llm_model, streaming=True) # 번역을 항상 같게 하기 위해서 설정

  parser = get_parser()
  format_instructions = parser.get_format_instructions()
  print(format_instructions)

  human_template="""\
    다음의 문장에서 다음과 같은 정보를 추출하라:

    gift: 다른 사람을 위한 선물로 구매했는가? \
          예인 경우 True, 그렇지 아닌경우는 False 값을 설정한다.

    delivery_days: 제품이 도착하는 데 도착하는 데 며칠이 걸렸나? \
                  이 정보를 찾을 수 없다면 -1 값을 설정한다.

    price_value: 값이나 가격에 대한 문장을 추출하여 쉼표로 구분된 Python 목록으로 출력한다.

    문장: {text}

    {format_instructions}
    """
  human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
  chat_prompt = ChatPromptTemplate.from_messages([human_message_prompt])

  chatchain = LLMChain(llm=chat, prompt=chat_prompt, verbose=True)

  text = """\
    이 잎 송풍기는 꽤 놀랍습니다.  촛불 송풍기, 산들바람, 바람이 부는 도시, 토네이도 등 네 가지 설정이 있습니다. \
    아내를 위해 결혼 기념일 선물로 샀는데, 딱 이틀 만에 도착했어요. \
    아내가 너무 좋아해서 말문이 막혔어요. \
    지금까지는 저 혼자만 사용하고 있으며 격일로 아침마다 잔디밭의 낙엽을 치우는 데 사용하고 있습니다. \
    다른 잎사귀 송풍기보다 약간 비싸지만 하지만 추가 기능을 생각하면 그만한 가치가 있다고 생각합니다. \
    """

  print_start()
  response = chatchain.run(text=text, format_instructions=format_instructions, callbacks=[handler])

  print_end()
  print('response')
  print(response)
  output_dict = parser.parse(response)
  print('output_dict')
  print(output_dict)

  print(output_dict.get('delivery_days'))


if __name__ == '__main__':
  main()