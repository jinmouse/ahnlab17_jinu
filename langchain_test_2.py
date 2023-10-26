import os
import openai

from dotenv import load_dotenv

from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain
from langchain.schema import BaseOutputParser

# openai.api_key = os.getenv("OPENAI_API_KEY")
# openai.organization = os.getenv("ORGANIZATION")

llm_model = "gpt-3.5-turbo"

load_dotenv()

class CommaSeparatedListOutputParser(BaseOutputParser):
    """LLM 아웃풋에 있는 ','를 분리해서 리턴하는 파서."""


    def parse(self, text: str):
        return text.strip().split(", ")

template = """
너는 5세 아이의 낱말놀이를 도와주는 AI야.
아이가 어떤 카테고리에 해당하는 개체들을 말해달라고 <질문>을 하면
해당 카테고리에 해당하는 단어들을 5개 나열해야 해.
이때 각 단어는 반드시 comma(,)로 분리해서 대답해주고, 이외의 말은 하지 마.

질문:"""
system_message_prompt = SystemMessagePromptTemplate.from_template(template)
human_template = "{text}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])


chain = LLMChain(
    llm=ChatOpenAI(temperature=0.0, model=llm_model),
    prompt=chat_prompt,
    output_parser=CommaSeparatedListOutputParser()
)
response = chain.run("동물에 대해 공부하고 싶어")
print(response)