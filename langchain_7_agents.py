#!/usr/bin/env python
# coding: utf-8

# # LangChain: Agents
#
# ## Outline:
#
# * Using built in LangChain tools: DuckDuckGo search and Wikipedia
# * Defining your own tools


#!pip install -U wikipedia

from dotenv import load_dotenv

llm_model = "gpt-3.5-turbo"

load_dotenv()



from langchain.agents.agent_toolkits import create_python_agent
from langchain.agents import load_tools, initialize_agent
from langchain.agents import AgentType
from langchain.tools.python.tool import PythonREPLTool
from langchain.python import PythonREPL
from langchain.chat_models import ChatOpenAI


llm = ChatOpenAI(temperature=0, model=llm_model)

# numexpr, wikipedia 라이브러리 추가
tools = load_tools(["llm-math","wikipedia"], llm=llm) 

def test_wikipedia() -> None:
  agent= initialize_agent(
    tools,
    llm,
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True,
    verbose = True)

  result = agent.run("페이스북 창업자는 누구인지? 그의 현재(2023년) 나이는? 그의 현재 나이를 제곱하면?")

  print(result)

  print(agent.tools)
  return None



# ## Python Agent

def test_python() -> None:
  agent = create_python_agent(
    llm,
    tool=PythonREPLTool(),
    verbose=True
  )

  customer_list = [["Harrison", "Chase"],
          ["Lang", "Chain"],
          ["Dolly", "Too"],
          ["Elle", "Elem"],
          ["Geoff","Fusion"],
          ["Trance","Former"],
          ["Jen","Ayai"]
          ]


  result = agent.run(f"""Sort these customers by \
  last name and then first name \
  and print the output: {customer_list}""")

  print(result)
  return None


import langchain

def test_python_debug() -> None:
  langchain.debug=True
  result = agent.run(f"""Sort these customers by \
  last name and then first name \
  and print the output: {customer_list}""")
  langchain.debug=False

  print(result)
  return None


# ## Define your own tool

#!pip install DateTime

from langchain.agents import tool
from datetime import date


@tool
def time(text: str) -> str:
  """Returns todays date, use this for any \
  questions related to knowing todays date. \
  The input should always be an empty string, \
  and this function will always return todays \
  date - any date mathmatics should occur \
  outside this function."""
  return str(date.today())


def test_my_own_tool() -> None:
  agent= initialize_agent(
    tools + [time],
    llm,
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True,
    verbose = True)

  try:
    result = agent("whats the date today?")
    print(result)
  except:
    print("exception on external access")

  return None

from langchain.agents import create_csv_agent

def test_csv() -> None:
  agent = create_csv_agent(
    ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613"),
    "./data/Data.csv",
    verbose=True,
    agent_type=AgentType.OPENAI_FUNCTIONS,
  )

  result = agent.run("how many rows are there?")
  print("result ==========> ")
  print(result)

  print("agent.tools ==========> ")
  print(agent.tools)
  return None




if __name__ == '__main__':
  # test_wikipedia()
  # test_python()
  # test_python_debug()
  # test_my_own_tool()
  # test_csv()

