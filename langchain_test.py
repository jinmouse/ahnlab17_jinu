from dotenv import load_dotenv

from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI

llm_model = "gpt-3.5-turbo"

llm = OpenAI(model=llm_model)
chat_model = ChatOpenAI(model=llm_model)

llm.predict("hi!")

