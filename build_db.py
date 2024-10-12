import tiktoken
import json
import glob
import os
# import latex2markdown
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain.document_loaders import UnstructuredHTMLLoader
from langchain.document_loaders import BSHTMLLoader

from dotenv import load_dotenv

load_dotenv()

model_name = "gpt-3.5-turbo-16k"

encoding = tiktoken.encoding_for_model(model_name)


def get_tiktoken_length(text):
    return len(encoding.encode(text))

# Build syllabus DB


loader = TextLoader('course_name.txt')
raw_documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=128,
    length_function=get_tiktoken_length,
    is_separator_regex=False,
)

documents = text_splitter.split_documents(raw_documents)

embedding = OpenAIEmbeddings(model="text-embedding-ada-002")

db = Chroma.from_documents(
    documents, embedding=embedding, persist_directory="chroma_db_course_name"
)


# # Build testbook DB

loader = PyPDFLoader("textbook.pdf")
pages = loader.load_and_split()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=128,
    length_function=get_tiktoken_length,
    is_separator_regex=False,
)

documents = text_splitter.split_documents(pages)

embedding = OpenAIEmbeddings(model="text-embedding-ada-002")

db = Chroma.from_documents(
    documents, embedding=embedding, persist_directory="chroma_db_textbook"
)


# Build course overview

loader = PyPDFLoader("course_overview.pdf")
pages = loader.load_and_split()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=128,
    length_function=get_tiktoken_length,
    is_separator_regex=False,
)

documents = text_splitter.split_documents(pages)

embedding = OpenAIEmbeddings(model="text-embedding-ada-002")

db = Chroma.from_documents(
    documents, embedding=embedding, persist_directory="chroma_db_course_overview"
)

# build Overview from html
# loader = BSHTMLLoader(
#     "Course Overview and Policies - Fall 2023_ Fall 2023-ECE 120-Introduction to Computing-Sections AD1, AD2, AD3, AD4, AD5, AD6, AD7, AD8, AD9, ADA, ADB, ADC, AL1, AL2, AL3, AL5.html")
# data = loader.load()

# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=512,
#     chunk_overlap=128,
#     length_function=get_tiktoken_length,
#     is_separator_regex=False,
# )

# documents = text_splitter.split_documents(data)

# embedding = OpenAIEmbeddings(model="text-embedding-ada-002")

# print(documents)
# db = Chroma.from_documents(
#     documents, embedding=embedding, persist_directory="chroma_db_courseoverview_policies"
# )


# build Office Hours from html
# loader = BSHTMLLoader(
#     "Course Overview and Policies - Fall 2023 Fall 2023-ECE 120-Introduction to Computing.html")
# data = loader.load()

# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=512,
#     chunk_overlap=128,
#     length_function=get_tiktoken_length,
#     is_separator_regex=False,
# )

# documents = text_splitter.split_documents(data)

# embedding = OpenAIEmbeddings(model="text-embedding-ada-002")

# print(documents)
# db = Chroma.from_documents(
#     documents, embedding=embedding, persist_directory="chroma_db_courseoverview_policies"
# )

# Build notebook DB

loader = PyPDFLoader("ece120-spring-2022-notes-for-students-1.pdf")
pages = loader.load_and_split()
# print(pages[0])

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=128,
    length_function=get_tiktoken_length,
    is_separator_regex=False,
)

documents = text_splitter.split_documents(pages)

embedding = OpenAIEmbeddings(model="text-embedding-ada-002")

db = Chroma.from_documents(
    documents, embedding=embedding, persist_directory="chroma_db_notes"
)


# Build from canvas

# directory_path = "./canvas"

# # Loop through all files in the directory
# for filename in os.listdir(directory_path):
#     file_path = os.path.join(directory_path, filename)
#     if os.path.isfile(file_path) and filename != ".DS_Store":

#         # print(filename)
#         if filename.endswith(".html"):
#             loader = UnstructuredHTMLLoader(file_path)
#             data = loader.load()
#         else:
#             loader = PyPDFLoader(file_path)
#             data = loader.load_and_split()

#         # if (filename == "Office Hours FA23.html"):
#         #     #print(data)
#         # if (filename.endswith(".pdf")):
#         #     print(data)
#         text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=512,
#             chunk_overlap=128,
#             length_function=get_tiktoken_length,
#             is_separator_regex=False,
#         )

#         documents = text_splitter.split_documents(data)

#         embedding = OpenAIEmbeddings(model="text-embedding-ada-002")

#         # print(documents)
#         filename = filename.replace(".html", "")
#         filename = filename.replace(".pdf", "")
#         filename = filename.replace(" ", "_")
#         db_name = "chroma_db_" + filename
#         print(db_name)
#         db = Chroma.from_documents(
#             documents, embedding=embedding, persist_directory=db_name
#         )
