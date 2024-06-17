import streamlit as st
from openai import OpenAI
import time
import os
import pickle
from langchain_text_splitters import CharacterTextSplitter
from unstructured.partition.pdf import partition_pdf

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

import base64
from langchain_core.messages import HumanMessage

import uuid

from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

import io
import re

from IPython.display import HTML, display
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from PIL import Image



OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]


st.title('재경업무 프로세스 질의응답')

def agent_message_generator(message):
    for word in message.split():
        yield word + " "
        time.sleep(0.05)

with st.chat_message("assistant"):
    st.write_stream(agent_message_generator("사용자의 질문에 응답하기 위해 준비중입니다."))
# with st.chat_message("assistant"):
#     st.write_stream(agent_message_generator("재경업무 프로세스 파일에서 컨텐츠 유형(텍스트, 이미지, 테이블)을 추출하겠습니다."))

# from openai import OpenAI

# # Set OpenAI API key from Streamlit secrets
# client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

fpath = "multi-modal/"
fname = "sample.pdf"

# PDF에서 요소 추출

# def extract_pdf_elements(path, fname):
#     """
#     PDF 파일에서 이미지, 테이블, 그리고 텍스트 조각을 추출합니다.
#     path: 이미지(.jpg)를 저장할 파일 경로
#     fname: 파일 이름
#     """
#     return partition_pdf(
#         filename=os.path.join(path, fname),
#         extract_images_in_pdf=True,  # PDF 내 이미지 추출 활성화
#         infer_table_structure=True,  # 테이블 구조 추론 활성화
#         chunking_strategy="by_title",  # 제목별로 텍스트 조각화
#         max_characters=4000,  # 최대 문자 수
#         new_after_n_chars=3800,  # 이 문자 수 이후에 새로운 조각 생성
#         combine_text_under_n_chars=2000,  # 이 문자 수 이하의 텍스트는 결합
#         image_output_dir_path=path,  # 이미지 출력 디렉토리 경로
#     )
# # 요소 추출
# with st.spinner('추출이 될때까지 잠시만 기다려 주세요....'):
#     raw_pdf_elements = extract_pdf_elements(fpath, fname)

# st.success('추출이 완료되었습니다.')

# # 데이터 저장
# with open('pk_raw_pdf_elements.pkl', "wb") as f:
#     pickle.dump(raw_pdf_elements, f, protocol=pickle.HIGHEST_PROTOCOL)


# 데이터 로드 - 저장된 pickle 객체를 불러서 진행
# with st.spinner('추출이 될때까지 잠시만 기다려 주세요....'):
#     with open('pk_raw_pdf_elements.pkl', "rb") as f:
#         raw_pdf_elements = pickle.load(f)
#     time.sleep(2)


# st.success('추출이 완료되었습니다.')

with open('pk_raw_pdf_elements.pkl', "rb") as f:
        raw_pdf_elements = pickle.load(f)



    # 요소를 유형별로 분류

def categorize_elements(raw_pdf_elements):
    """
    PDF에서 추출된 요소를 테이블과 텍스트로 분류합니다.
    raw_pdf_elements: unstructured.documents.elements의 리스트
    """
    tables = []  # 테이블 저장 리스트
    texts = []  # 텍스트 저장 리스트
    for element in raw_pdf_elements:
        if "unstructured.documents.elements.Table" in str(type(element)):
            tables.append(str(element))  # 테이블 요소 추가
        elif "unstructured.documents.elements.CompositeElement" in str(type(element)):
            texts.append(str(element))  # 텍스트 요소 추가
    return texts, tables


# with st.chat_message("assistant"):
#     st.write_stream(agent_message_generator("다음은 PDF에서 추출된 요소를 테이블과 텍스트로 분류하겠습니다."))

# 텍스트, 테이블 추출
# with st.spinner('추출이 될때까지 잠시만 기다려 주세요....'):
#     texts, tables = categorize_elements(raw_pdf_elements)
#     # print("texts 추출 완료: ", texts)
#     # print("tables 추출 완료: ", texts)

#     # 선택사항: 텍스트에 대해 특정 토큰 크기 적용
#     text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
#         chunk_size=1000, chunk_overlap=0  # 텍스트를 4000 토큰 크기로 분할, 중복 없음
#     )
#     joined_texts = " ".join(texts)  # 텍스트 결합
#     texts_4k_token = text_splitter.split_text(joined_texts)  # 분할 실행

# st.success('추출이 완료되었습니다.')


texts, tables = categorize_elements(raw_pdf_elements)
# print("texts 추출 완료: ", texts)
# print("tables 추출 완료: ", texts)

# 선택사항: 텍스트에 대해 특정 토큰 크기 적용
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=800, chunk_overlap=0  # 텍스트를 4000 토큰 크기로 분할, 중복 없음
)
joined_texts = " ".join(texts)  # 텍스트 결합
texts_4k_token = text_splitter.split_text(joined_texts)  # 분할 실행





def generate_text_summaries(texts, tables, summarize_texts=False):
    """
    텍스트 요소 요약
    texts: 문자열 리스트
    tables: 문자열 리스트
    summarize_texts: 텍스트 요약 여부를 결정. True/False
    """

    # 프롬프트 설정
    prompt_text = """You are an assistant tasked with summarizing tables and text for retrieval. \
    These summaries will be embedded and used to retrieve the raw text or table elements. \
    Give a concise summary of the table or text that is well optimized for retrieval. Table or text: {element} """
    prompt = ChatPromptTemplate.from_template(prompt_text)

    # 텍스트 요약 체인
    model = ChatOpenAI(temperature=0, model="gpt-4o", api_key=OPENAI_API_KEY)
    summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()

    # 요약을 위한 빈 리스트 초기화
    text_summaries = []
    table_summaries = []

    # 제공된 텍스트에 대해 요약이 요청되었을 경우 적용
    if texts and summarize_texts:
        text_summaries = summarize_chain.batch(texts, {"max_concurrency": 5})
    elif texts:
        text_summaries = texts

    # 제공된 테이블에 적용
    if tables:
        table_summaries = summarize_chain.batch(tables, {"max_concurrency": 5})

    return text_summaries, table_summaries

# with st.chat_message("assistant"):
#     st.write_stream(agent_message_generator("다음은 텍스트와 테이블에 대한 요약을 생성하겠습니다."))

# 텍스트, 테이블 요약 가져오기
# with st.spinner('요약문이 작성될 때까지 잠시만 기다려 주세요....'):
#     text_summaries, table_summaries = generate_text_summaries(
#         texts_4k_token, tables, summarize_texts=True
#     )

# # # 데이터 저장
# with open('pk_text_summaries.pkl', "wb") as f:
#     pickle.dump(text_summaries, f, protocol=pickle.HIGHEST_PROTOCOL)

# with open('pk_table_summaries.pkl', "wb") as f:
#     pickle.dump(table_summaries, f, protocol=pickle.HIGHEST_PROTOCOL)

# 데이터 로드 - 저장된 pickle 객체를 불러서 진행
# with st.spinner('추출이 될때까지 잠시만 기다려 주세요....'):
#     with open('pk_text_summaries.pkl', "rb") as f:
#         text_summaries = pickle.load(f)    

#     with open('pk_table_summaries.pkl', "rb") as f:
#         table_summaries = pickle.load(f) 

# st.success('테트스와 테이블에 대한 요약문이 생성되었습니다.')


with open('pk_text_summaries.pkl', "rb") as f:
    text_summaries = pickle.load(f)    

with open('pk_table_summaries.pkl', "rb") as f:
    table_summaries = pickle.load(f) 





######################### 이미지 요약문 생성 #########################

# with st.chat_message("assistant"):
#     st.write_stream(agent_message_generator("다음은 'GPT-4o' 모델을 사용하여 각 이미지에 대해 요약문(summary)를 생성하겠습니다.."))

def encode_image(image_path):
    # 이미지 파일을 base64 문자열로 인코딩합니다.
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def image_summarize(img_base64, prompt):
    # 이미지 요약을 생성합니다.
    chat = ChatOpenAI(model="gpt-4o", max_tokens=2048, api_key=OPENAI_API_KEY)

    msg = chat.invoke(
        [
            HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"},
                    },
                ]
            )
        ]
    )
    return msg.content

def generate_img_summaries(path):
    """
    이미지에 대한 요약과 base64 인코딩된 문자열을 생성합니다.
    path: Unstructured에 의해 추출된 .jpg 파일 목록의 경로
    """

    # base64로 인코딩된 이미지를 저장할 리스트
    img_base64_list = []

    # 이미지 요약을 저장할 리스트
    image_summaries = []

    # 요약을 위한 프롬프트
    prompt = """You are an assistant tasked with summarizing images for retrieval. \
    These summaries will be embedded and used to retrieve the raw image. \
    Give a concise summary of the image that is well optimized for retrieval."""

    # 이미지에 적용
    for img_file in sorted(os.listdir(path)):
        if img_file.endswith(".jpg"):
            img_path = os.path.join(path, img_file)
            base64_image = encode_image(img_path)
            img_base64_list.append(base64_image)
            image_summaries.append(image_summarize(base64_image, prompt))

    return img_base64_list, image_summaries

# 이미지 요약 실행

# with st.spinner('이미지 요약문 생성이 완료될 때까지 기다려 주세요....'):
#     fpath = "figures/"
#     img_base64_list, image_summaries = generate_img_summaries(fpath)

# # 데이터 저장
# with open('pk_img_base64_list.pkl', "wb") as f:
#     pickle.dump(img_base64_list, f, protocol=pickle.HIGHEST_PROTOCOL)

# with open('pk_image_summaries.pkl', "wb") as f:
#     pickle.dump(image_summaries, f, protocol=pickle.HIGHEST_PROTOCOL)

# st.success('추출된 모든 이미지에 대해 이미지 요약문 생성이 완료되었습니다.')

# 데이터 로드 - 저장된 pickle 객체를 불러서 진행
with st.spinner('이미지 요약문 생성이 완료될 때까지 기다려 주세요....'):
    with open('pk_img_base64_list.pkl', "rb") as f:
        img_base64_list = pickle.load(f)
    with open('pk_image_summaries.pkl', "rb") as f:
        image_summaries = pickle.load(f)    

# st.success('추출된 모든 이미지에 대해 이미지 요약문 생성이 완료되었습니다.')    


# with st.chat_message("assistant"):
#     st.write_stream(agent_message_generator("다음은 원본 텍스트, 이미지, 테이블을 검색할 수 있게 멀티검색기를 생성하겠습니다."))

def create_multi_vector_retriever(
    vectorstore, text_summaries, texts, table_summaries, tables, image_summaries, images
):
    """
    요약을 색인화하지만 원본 이미지나 텍스트를 반환하는 검색기를 생성합니다.
    """

    # 저장 계층 초기화
    store = InMemoryStore()
    id_key = "doc_id"

    # 멀티 벡터 검색기 생성
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=store,
        id_key=id_key,
    )

    # 문서를 벡터 저장소와 문서 저장소에 추가하는 헬퍼 함수
    def add_documents(retriever, doc_summaries, doc_contents):
        doc_ids = [
            str(uuid.uuid4()) for _ in doc_contents
        ]  # 문서 내용마다 고유 ID 생성
        summary_docs = [
            Document(page_content=s, metadata={id_key: doc_ids[i]})
            for i, s in enumerate(doc_summaries)
        ]
        retriever.vectorstore.add_documents(
            summary_docs
        )  # 요약 문서를 벡터 저장소에 추가
        retriever.docstore.mset(
            list(zip(doc_ids, doc_contents))
        )  # 문서 내용을 문서 저장소에 추가

    # 텍스트, 테이블, 이미지 추가
    if text_summaries:
        add_documents(retriever, text_summaries, texts)

    if table_summaries:
        add_documents(retriever, table_summaries, tables)

    if image_summaries:
        add_documents(retriever, image_summaries, images)

    return retriever

vectorstore = Chroma(
    collection_name="sample-rag-multi-modal", embedding_function=OpenAIEmbeddings(api_key=OPENAI_API_KEY)
)

# 검색기 생성
retriever_multi_vector_img = create_multi_vector_retriever(
    vectorstore,
    text_summaries,
    texts,
    table_summaries,
    tables,
    image_summaries,
    img_base64_list,
)

# st.success('멀티 벡터 검색기가 생성되었습니다..')  

# with st.chat_message("assistant"):
#     st.write_stream(agent_message_generator("RAG 검색기를 생성하여 사용자의 질문에 이미지 및 텍스트를 포함하여 답변할 수 있게 합니다.."))
# with st.chat_message("assistant"):
#     st.write_stream(agent_message_generator("멀티모달 RAG 체인을 구성중입니다. 잠시만 기다려주세요."))


def plt_img_base64(img_base64):
    """base64 인코딩된 문자열을 이미지로 표시"""
    # base64 문자열을 소스로 사용하는 HTML img 태그 생성
    image_html = f'<img src="data:image/jpeg;base64,{img_base64}" />'
    # HTML을 렌더링하여 이미지 표시
    display(HTML(image_html))


def looks_like_base64(sb):
    """문자열이 base64로 보이는지 확인"""
    return re.match("^[A-Za-z0-9+/]+[=]{0,2}$", sb) is not None


def is_image_data(b64data):
    """
    base64 데이터가 이미지인지 시작 부분을 보고 확인
    """
    image_signatures = {
        b"\xff\xd8\xff": "jpg",
        b"\x89\x50\x4e\x47\x0d\x0a\x1a\x0a": "png",
        b"\x47\x49\x46\x38": "gif",
        b"\x52\x49\x46\x46": "webp",
    }
    try:
        header = base64.b64decode(b64data)[:8]  # 처음 8바이트를 디코드하여 가져옴
        for sig, format in image_signatures.items():
            if header.startswith(sig):
                return True
        return False
    except Exception:
        return False
    
def resize_base64_image(base64_string, size=(128, 128)):
    """
    Base64 문자열로 인코딩된 이미지의 크기 조정
    """
    # Base64 문자열 디코드
    img_data = base64.b64decode(base64_string)
    img = Image.open(io.BytesIO(img_data))

    # 이미지 크기 조정
    resized_img = img.resize(size, Image.LANCZOS)

    # 조정된 이미지를 바이트 버퍼에 저장
    buffered = io.BytesIO()
    resized_img.save(buffered, format=img.format)

    # 조정된 이미지를 Base64로 인코딩
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def split_image_text_types(docs):
    """
    base64로 인코딩된 이미지와 텍스트 분리
    """
    b64_images = []
    texts = []
    for doc in docs:
        # 문서가 Document 타입인 경우 page_content 추출
        if isinstance(doc, Document):
            doc = doc.page_content
        if looks_like_base64(doc) and is_image_data(doc):
            doc = resize_base64_image(doc, size=(1300, 600))
            b64_images.append(doc)
        else:
            texts.append(doc)
    return {"images": b64_images, "texts": texts}

def img_prompt_func(data_dict):
    """
    컨텍스트를 단일 문자열로 결합
    """
    formatted_texts = "\n".join(data_dict["context"]["texts"])
    messages = []

    # 이미지가 있으면 메시지에 추가
    if data_dict["context"]["images"]:
        for image in data_dict["context"]["images"]:
            image_message = {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image}"},
            }
            messages.append(image_message)

    # 분석을 위한 텍스트 추가
    text_message = {
        "type": "text",
        "text": (            
            # "You are a employee of financial department in Hyundai Motors tasking with providing FAQ service about R&D.\n"
            # "You will be given a mixed of text, tables, and image(s) usually of charts or graphs.\n"
            # "Use this information to provide investment advice related to the user question. Answer in Korean. Do NOT translate company names.\n"
            "You are working in the finance department at Hyundai Motor Research Institute in Korea. When Hyundai Motor researchers ask you about finance tasks such as budget execution, voucher processing, and corporate card use, you answer their inquiries.\n"
            "You will be given a mixed of text, tables, and image(s) about process manual using SAP Systems\n"
            "When you receive questions from users (researchers) about finance tasks such as budget execution, voucher processing, and corporate card use, please respond with text, images, and tables. And please provide your answers in Korean.\n"
            "\n"
            "Return the results with the most relevant image found by the multi-vector retriever as the first image. Especially for images, find the single most relevant image.\n"            
            f"User-provided question: {data_dict['question']}\n\n"
            "Text and / or table:\n"
            f"{formatted_texts}"
        ),
    }
    messages.append(text_message)
    return [HumanMessage(content=messages)]

def multi_modal_rag_chain(retriever):
    """
    멀티모달 RAG 체인
    """

    # 멀티모달 LLM
    # model = ChatOpenAI(temperature=0, model="gpt-4o", max_tokens=2048)
    model = ChatOpenAI(temperature=0, model="gpt-4o", max_tokens=1024)

    # RAG 파이프라인
    chain = (
        {
            "context": retriever | RunnableLambda(split_image_text_types),
            "question": RunnablePassthrough(),
        }
        | RunnableLambda(img_prompt_func)
        | model
        | StrOutputParser()
    )

    return chain

# RAG 체인 생성
chain_multimodal_rag = multi_modal_rag_chain(retriever_multi_vector_img)

# with st.chat_message("assistant"):
#     st.write_stream(agent_message_generator("기다려 주셔서 감사합니다. 사전 작업이 완료되어 지금부터 재경 관련 질문을 주시면 성심히 답변드리겠습니다."))

# with st.chat_message("assistant"):
#     st.write_stream(agent_message_generator("질문내용에 대한 답변은 텍스트와 이미지가 혼합된 형태로 제공됩니다.단, 관련 이미지가 없는 경우는 텍스트만 답변으로 출력됩니다."))

st.balloons()
time.sleep(1.3)


# query = st.text_input("재경업무 프로세스에 대해 어느 것이든 물어봐주세요.")
queries = [
    '전표 변경 시에 개별 항목 입력 방법을 알래줄래?',
    # '전표 변경 시에 추가 데이터 입력방법을 알려줄래?',
    '연간보고서 작성 절차에 대해 설명해줘',
    '정부과제 전표처리 시 주의사항은 뭐야?'   
]



for query in queries:
    if query:        
        
        docs = retriever_multi_vector_img.invoke(query, limit=6)
        
        with st.chat_message("user"):
            st.write(query)

        with st.status("답변을 생성하고 있습니다.", expanded=True):
            with st.chat_message("assistant"):
                response = chain_multimodal_rag.invoke(query)
                for i, doc in enumerate(docs):
                    # if looks_like_base64(doc) & is_image_data(doc):
                    #     st.write(f'<img src="data:image/jpeg;base64,{docs[0]}"  width="600" height="400" />', unsafe_allow_html=True)                        
                    #     break
                    st.write(response, unsafe_allow_html=True)
                    if looks_like_base64(doc):
                        st.write(f'<img src="data:image/jpeg;base64,{docs[i]}"  width="600" height="400" />', unsafe_allow_html=True)
            
                # st.write(f"관련 이미지의 수는 {len(docs)} 개입니다.")



# mapping_table = {
#     0: img_base64_list[0],
#     1: img_base64_list[1],
#     2: img_base64_list[2],
#     3: img_base64_list[3],
#     4: img_base64_list[4],
# }

# for query in queries:
#     if query:        
        
#         for i in mapping_table.keys():
#             st.write(f'<img src="data:image/jpeg;base64,{mapping_table[i]}" />', unsafe_allow_html=True)

# for i in range(len(img_base64_list)):
#     st.write(str(i)+ "번째 이미지")
#     st.write(f'<img src="data:image/jpeg;base64,{img_base64_list[i]}" />', unsafe_allow_html=True)
