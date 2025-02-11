from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama

# Load the PDF file
pdf_path = "/Users/rafsanmallik/PycharmProjects/pythonProject8/kid.pdf"
loader = PyMuPDFLoader(pdf_path)
documents = loader.load()

# Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
all_splits = text_splitter.split_documents(documents)

# Print the split text chunks
for split in all_splits:
    print(split.page_content)

local_embeddings = OllamaEmbeddings(model="nomic-embed-text")

vectorstore = Chroma.from_documents(documents=all_splits, embedding=local_embeddings)

question = "What are the approaches to Task Decomposition?"
docs = vectorstore.similarity_search(question)



model = ChatOllama(
    model="deepseek-r1:1.5b",
)

response_message = model.invoke(
    "What is Kidney disease ?"
)

print(response_message.content)
