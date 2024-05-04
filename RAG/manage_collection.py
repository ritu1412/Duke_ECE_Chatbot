import chromadb.utils.embedding_functions as embedding_functions
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import chromadb
import argparse
import os


load_dotenv()
hf_token = os.getenv("HF_READ_TOKEN")
openai_token = os.getenv("OPENAI_TOKEN")

def extract_from_pdf(pdf_doc):
    pdf_reader = PdfReader(pdf_doc)
    text = ''
    ## now loop through the pages in the pdf
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def get_text_chunks(filepath):
    ## returns a single string with all the raw text from pdfs
    text = ""
    files = os.listdir(filepath)
    for file in files:
        if file.endswith('.pdf'):
            text += extract_from_pdf(os.path.join(filepath,file))
        if file.endswith('.txt'):
            with open(os.path.join(filepath,file),'r') as txtfile:
                text += txtfile.read()
    
    ## create a new instance
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)

    chunks = text_splitter.split_text(text)  ## returns a list of chunks with each chunk size 100

    return chunks

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('ACTION',type=str,help='valid options: NEW,DELETE,ADD')
    parser.add_argument('C_NAME',type=str,help='Name of the collection')
    parser.add_argument('FILE_PATH',nargs='?',type=str,default=None,help='file path')
    args = parser.parse_args()
    action = args.ACTION
    c_name = args.C_NAME
    file_path = args.FILE_PATH
    print(action,c_name,file_path)

    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key=openai_token,
                model_name="text-embedding-ada-002")
    
    ragclient = chromadb.PersistentClient(path="DB/")
        
    if action != "DELETE":
        if file_path is None:
            raise ValueError("File path needed when adding documents to NEW or EXISTING collection!")
        
        if action == "NEW":
            collection = ragclient.create_collection(name=c_name,embedding_function=openai_ef)
        else:
            collection = ragclient.get_collection(name=c_name,embedding_function=openai_ef)
        text_chunks = get_text_chunks(file_path)
        collection.add(documents=text_chunks,ids=[str(i) for i in range(1,len(text_chunks)+1)])
    else:
        ragclient.delete_collection(name=c_name)
    

if __name__ == "__main__":
    main()

    
