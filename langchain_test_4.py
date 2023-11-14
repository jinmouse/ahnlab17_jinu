# 필요한 라이브러리 설치
# !pip install langchain pdfplumber

# 라이브러리 임포트
import pdfplumber
from langchain.embeddings import OpenAIEmbeddings
from langchain.textsplitters import CharacterTextSplitter
from langchain.vectorstores import VectorStore

# PDF 파일을 텍스트로 변환하는 함수
def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

# 텍스트를 임베딩하고 벡터 스토어에 저장하는 함수
def embed_text_to_vector_store(text, embeddings, vector_store):
    # 텍스트 분할
    splitter = CharacterTextSplitter(max_length=4000)
    parts = splitter.split(text)

    # 각 부분에 대해 임베딩 생성 및 저장
    for part in parts:
        embedding = embeddings.embed(part)
        vector_store.add(embedding)

# 요약 생성 함수
def generate_summary(vector_store):
    # 벡터 스토어에서 데이터를 사용하여 요약 생성
    # 이 부분은 langchain의 요약 생성 기능에 따라 다를 수 있음
    # 예시 코드는 상징적인 것으로 실제 API와 다를 수 있음
    summary = vector_store.create_summary()
    return summary

# 메인 프로그램
def main(pdf_path):
    # OpenAI 임베딩 초기화
    embeddings = OpenAIEmbeddings()

    # 벡터 스토어 초기화
    vector_store = VectorStore()

    # PDF 파일에서 텍스트 추출
    text = extract_text_from_pdf(pdf_path)

    # 텍스트 임베딩 및 벡터 스토어에 저장
    embed_text_to_vector_store(text, embeddings, vector_store)

    # 요약 생성
    summary = generate_summary(vector_store)
    return summary

# PDF 파일 경로
pdf_path = "your_pdf_file.pdf"

# 요약 실행
print(main(pdf_path))
