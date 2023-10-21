import time

from dotenv import load_dotenv


from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts.chat import (
  ChatPromptTemplate,
  SystemMessagePromptTemplate,
  HumanMessagePromptTemplate,
)

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


def main() -> None:
  chat = ChatOpenAI(temperature=0, model=llm_model) # 번역을 항상 같게 하기 위해서 설정

  template="You are a helpful assisstant that tranlates {input_language} to {output_language}."
  system_message_prompt = SystemMessagePromptTemplate.from_template(template)

  human_template="{text}"
  human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

  chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

  chatchain = LLMChain(llm=chat, prompt=chat_prompt, verbose=True)
  text = """
Exclusive: US will transfer weapons seized from Iran to Ukraine


The US will transfer thousands of seized Iranian weapons and rounds of ammunition to Ukraine, in a move that could help to alleviate some of the critical shortages facing the Ukrainian military as it awaits more money and equipment from the US and its allies, US officials said.
"""

  print_start()
  response = chatchain.run(input_language="English", output_language="Korean", text=text)

  print_end()

  print(response)


if __name__ == '__main__':
  main()