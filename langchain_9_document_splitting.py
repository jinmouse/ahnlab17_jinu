#!/usr/bin/env python
# coding: utf-8

# # Document Splitting


import os
import time
import json

import openai

from dotenv import load_dotenv

load_dotenv()


openai.api_key = os.getenv("OPENAI_API_KEY")
openai.organization = os.getenv("ORGANIZATION")


from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter


# ## Recursive splitting details
#
# `RecursiveCharacterTextSplitter` is recommended for generic text.


some_text = """When writing documents, writers will use document structure to group content. \
This can convey to the reader, which idea's are related. For example, closely related ideas \
are in sentances. Similar ideas are in paragraphs. Paragraphs form a document. \n\n  \
Paragraphs are often delimited with a carriage return or two carriage returns. \
Carriage returns are the "backslash n" you see embedded in this string. \
Sentences have a period at the end, but also, have a space.\
and words are separated by space."""


def test_text_split() -> None:
  print(f"len(some_text)=>{len(some_text)}")

  c_splitter = CharacterTextSplitter(
    chunk_size=450,
    chunk_overlap=0,
    separator = ' '
  )
  r_splitter = RecursiveCharacterTextSplitter(
    chunk_size=450,
    chunk_overlap=0,
    separators=["\n\n", "\n", " ", ""]
  )
  c_split_result = c_splitter.split_text(some_text)
  r_split_result = r_splitter.split_text(some_text)

  print(f"c_split_result=>{c_split_result}")
  print(f"r_split_result=>{r_split_result}")
  return None



# Let's reduce the chunk size a bit and add a period to our separators:

def test_text_small_chunk() -> None:
  r_splitter = RecursiveCharacterTextSplitter(
    chunk_size=150,
    chunk_overlap=0,
    separators=["\n\n", "\n", "\. ", " ", ""]
  )
  r_result_150 = r_splitter.split_text(some_text)
  print(f"r_result_150=>{r_result_150}")

  r_splitter = RecursiveCharacterTextSplitter(
    chunk_size=150,
    chunk_overlap=0,
    separators=["\n\n", "\n", "(?<=\. )", " ", ""]
  )
  r_result_150_2 = r_splitter.split_text(some_text)
  print(f"r_result_150_2=>{r_result_150_2}")
  return None

from langchain.document_loaders import PyPDFLoader

def test_pdf_split() -> None:
  loader = PyPDFLoader("./data/프리랜서 가이드라인 (출판본).pdf")
  global pages
  pages = loader.load()
  text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=1000,
    chunk_overlap=150,
    length_function=len
  )
  docs = text_splitter.split_documents(pages)
  print(f"len(docs)=>{len(docs)}")
  print(f"len(pages)=>{len(pages)}")
  return None



from langchain.text_splitter import TokenTextSplitter

def test_token_split() -> None:
  text_splitter = TokenTextSplitter(chunk_size=1, chunk_overlap=0)
  text1 = "foo bar bazzyfoo"
  result = text_splitter.split_text(text1)
  print(f"result=>{result}")

  text_splitter = TokenTextSplitter(chunk_size=10, chunk_overlap=0)
  docs = text_splitter.split_documents(pages)
  print(f"docs[0]=>{docs[0]}")
  print(f"pages[0].metadata=>{pages[0].metadata}")
  return None



# ## Markdown splitting


from langchain.text_splitter import MarkdownHeaderTextSplitter

def test_markdown() -> None:
  markdown_document = """# Title\n\n \
  ## Chapter 1\n\n \
  Hi this is Jim\n\n Hi this is Joe\n\n \
  ### Section \n\n \
  Hi this is Lance \n\n
  ## Chapter 2\n\n \
  Hi this is Molly"""

  headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
  ]


  markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on
  )
  md_header_splits = markdown_splitter.split_text(markdown_document)
  print(f"md_header_splits[0]=>{md_header_splits[0]}")
  print(f"md_header_splits[1]=>{md_header_splits[1]}")
  return None




if __name__ == '__main__':
  test_text_split()
  test_text_small_chunk()
  test_pdf_split()
  test_token_split()
  test_markdown()