#!/usr/bin/env python
# coding: utf-8

# # Chains in LangChain
#
# ## Outline
#
# * LLMChain
# * Sequential Chains
#   * SimpleSequentialChain
#   * SequentialChain



#!pip install pandas

from dotenv import load_dotenv

llm_model = "gpt-3.5-turbo"

load_dotenv()

import pandas as pd

# ## LLMChain

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

def test_LLMChain() -> None:
  llm = ChatOpenAI(temperature=0.9, model=llm_model)

  prompt = ChatPromptTemplate.from_template(
    "'{product}'을 만드는 회사를 나타내는 최고로 좋은 이름 하나를 추천하라"
  )

  chain = LLMChain(llm=llm, prompt=prompt)

  global product
  product = "퀸 사이즈 침대 시트 세트"
  print(chain.run(product))
  return None



# ## SimpleSequentialChain

from langchain.chains import SimpleSequentialChain

def test_SimpleSequentialChain() -> None:
  llm = ChatOpenAI(temperature=0.9, model=llm_model)

  # prompt template 1
  first_prompt = ChatPromptTemplate.from_template(
    "'{product}'을 만드는 회사를 나타내는 최고로 좋은 이름 하나를 추천하라"
  )

  # Chain 1
  chain_one = LLMChain(llm=llm, prompt=first_prompt)

  # prompt template 2
  second_prompt = ChatPromptTemplate.from_template(
    "다음 회사를 20단어 이내로 표현하라:{company_name}"
  )
  # chain 2
  chain_two = LLMChain(llm=llm, prompt=second_prompt)


  overall_simple_chain = SimpleSequentialChain(chains=[chain_one, chain_two],
                        verbose=True
                        )

  response = overall_simple_chain.run(product)
  print(response)
  return None


# ## SequentialChain

from langchain.chains import SequentialChain

def test_SequentialChain(df: pd.DataFrame) -> None:
  llm = ChatOpenAI(temperature=0.9, model=llm_model)

  # prompt template 1: translate to english
  first_prompt = ChatPromptTemplate.from_template(
    "다음 리뷰를 한국어로 번역하라:"
    "\n\n{Review}"
  )
  # chain 1: input= Review and output= Korean_Review
  chain_one = LLMChain(llm=llm, prompt=first_prompt,
            output_key="Korean_Review"
            )

  second_prompt = ChatPromptTemplate.from_template(
    "다음 리뷰를 1문장으로 요약하라:"
    "\n\n{Korean_Review}"
  )
  # chain 2: input= Korean_Review and output= summary
  chain_two = LLMChain(llm=llm, prompt=second_prompt,
            output_key="summary"
            )


  # prompt template 3: translate to english
  third_prompt = ChatPromptTemplate.from_template(
    "다음 리뷰의 언어는 무엇인가:\n\n{Review}"
  )
  # chain 3: input= Review and output= language
  chain_three = LLMChain(llm=llm, prompt=third_prompt,
              output_key="language"
              )


  # prompt template 4: follow up message
  fourth_prompt = ChatPromptTemplate.from_template(
    "다음의 요약 내용에 대해서 주어진 언어로 후속 답변을 작성하라:"
    "\n\n요약: {summary}\n\n언어: {language}"
  )
  # chain 4: input= summary, language and output= followup_message
  chain_four = LLMChain(llm=llm, prompt=fourth_prompt,
              output_key="followup_message"
            )

  # overall_chain: input= Review
  # and output= Korean_Review,summary, followup_message
  overall_chain = SequentialChain(
    chains=[chain_one, chain_two, chain_three, chain_four],
    input_variables=["Review"],
    output_variables=["Korean_Review", "summary","followup_message"],
    verbose=True
  )

  review = df.Review[5]
  response = overall_chain(review)
  print(response)
  return None


if __name__ == '__main__':
  df: pd.DataFrame = pd.read_csv('data/Data.csv')
  print(df.head())

  test_LLMChain()
  test_SimpleSequentialChain()
  test_SequentialChain(df)