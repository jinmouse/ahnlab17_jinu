#!/usr/bin/env python
# coding: utf-8

# # LangChain: Memory
#
# ## Outline
# * ConversationBufferMemory
# * ConversationBufferWindowMemory
# * ConversationTokenBufferMemory
# * ConversationSummaryMemory

#pip install --upgrade langchain

import os
import time
import openai
from dotenv import load_dotenv
load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
openai.organization = os.getenv("ORGANIZATION")
llm_model = "gpt-3.5-turbo"

from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory




# ## ConversationBufferMemory


def test_ConversationBufferMemory() -> None:
  llm = ChatOpenAI(temperature=0.0, model=llm_model)
  memory = ConversationBufferMemory()
  conversation = ConversationChain(
    llm=llm,
    memory = memory,
    verbose=True
  )
  # prompt 입력 1
  print(conversation.predict(input="Hi, my name is Andrew"))
  # prompt 입력 2
  print(conversation.predict(input="What is 1+1?"))
  # prompt 입력 3
  print(conversation.predict(input="What is my name?"))
  print(memory.buffer)
  print(memory.load_memory_variables({}))

  # prompt 입력없이 저장하기
  memory = ConversationBufferMemory()
  memory.save_context({"input": "Hi"},
                    {"output": "What's up"})
  print(memory.buffer)
  print(memory.load_memory_variables({}))
  memory.save_context({"input": "Not much, just hanging"},
                    {"output": "Cool"})
  print(memory.load_memory_variables({}))
  return None



# ## ConversationBufferWindowMemory

from langchain.memory import ConversationBufferWindowMemory

def test_ConversationBufferWindowMemory() -> None:
  memory = ConversationBufferWindowMemory(k=1)
  memory.save_context({"input": "Hi"},
                      {"output": "What's up"})
  memory.save_context({"input": "Not much, just hanging"},
                      {"output": "Cool"})
  print(memory.load_memory_variables({}))

  llm = ChatOpenAI(temperature=0.0, model=llm_model)
  memory = ConversationBufferWindowMemory(k=1)
  conversation = ConversationChain(
    llm=llm,
    memory = memory,
    verbose=False
  )
  print(conversation.predict(input="Hi, my name is Andrew"))
  print(conversation.predict(input="What is 1+1?"))
  print(conversation.predict(input="What is my name?"))
  return None



# ## ConversationTokenBufferMemory

from langchain.memory import ConversationTokenBufferMemory

def test_ConversationTokenBufferMemory() -> None:
  llm = ChatOpenAI(temperature=0.0, model=llm_model)

  memory = ConversationTokenBufferMemory(llm=llm, max_token_limit=50)
  memory.save_context({"input": "AI is what?!"},
                      {"output": "Amazing!"})
  memory.save_context({"input": "Backpropagation is what?"},
                      {"output": "Beautiful!"})
  memory.save_context({"input": "Chatbots are what?"},
                      {"output": "Charming!"})

  print(memory.load_memory_variables({}))
  return None




# ## ConversationSummaryMemory

from langchain.memory import ConversationSummaryBufferMemory

def test_ConversationSummaryBufferMemory() -> None:
  llm = ChatOpenAI(temperature=0.0, model=llm_model)
  # create a long string
  schedule = "There is a meeting at 8am with your product team. \
  You will need your powerpoint presentation prepared. \
  9am-12pm have time to work on your LangChain \
  project which will go quickly because Langchain is such a powerful tool. \
  At Noon, lunch at the italian resturant with a customer who is driving \
  from over an hour away to meet you to understand the latest in AI. \
  Be sure to bring your laptop to show the latest LLM demo."

  memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=100)
  memory.save_context({"input": "Hello"}, {"output": "What's up"})
  memory.save_context({"input": "Not much, just hanging"},
                      {"output": "Cool"})
  memory.save_context({"input": "What is on the schedule today?"},
                      {"output": f"{schedule}"})
  print(memory.load_memory_variables({}))

  conversation = ConversationChain(
    llm=llm,
    memory = memory,
    verbose=True
  )

  print(conversation.predict(input="What would be a good demo to show?"))
  print(memory.load_memory_variables({}))
  return None


def _title(title: str) -> None:
  print(f" \
        \n================================================= \
        \n{title} \
        \n=================================================\n \
        ")


if __name__ == '__main__':
  _title("test_ConversationBufferMemory()")
  test_ConversationBufferMemory()

  _title("test_ConversationBufferWindowMemory()")
  test_ConversationBufferWindowMemory()

  _title("test_ConversationTokenBufferMemory()")
  test_ConversationTokenBufferMemory()

  _title("test_ConversationSummaryBufferMemory()")
  test_ConversationSummaryBufferMemory()

