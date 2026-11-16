import os
import glob
import time
import pymupdf4llm
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_aws import BedrockEmbeddings
from langchain_community.vectorstores import FAISS

# 1. Configuration constants
DATA_FOLDER = "data"
VECTOR_DB_PATH = "faiss_index"
BEDROCK_REGION = "ap-southeast-2"  # Matches your configured AWS region

def load_and_parse_pdfs(folder_path):
    """
    Reads every PDF in data/ page-by-page using PyMuPDF4LLM
    so financial/insurance tables are converted into clean Markdown tables.
    """
    all_documents = []
    pdf_files = sorted(glob.glob(os.path.join(folder_path, "*.pdf")))
    
    print(f"Found {len(pdf_files)} PDF files. Starting table-aware extraction...\n")
    
    for pdf_path in pdf_files:
        filename = os.path.basename(pdf_path)
        print(f"Processing: {filename}...")
        
        pages_data = pymupdf4llm.to_markdown(pdf_path, page_chunks=True)
        
        for page_num, page in enumerate(pages_data, start=1):
            page_text = page.get("text", "")
            if not page_text.strip():
                continue
                
            doc = Document(
                page_content=page_text,
                metadata={
                    "source": filename,
                    "page": page_num
                }
            )
            all_documents.append(doc)
            
    print(f"\nSuccessfully extracted {len(all_documents)} pages across all PDFs.")
    return all_documents

def split_documents(documents):
    """
    Splits extracted pages into smaller overlapping chunks so vector search is accurate.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n## ", "\n### ", "\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} searchable chunks.")
    return chunks

def create_vector_store(chunks, save_path):
    """
    Calls AWS Bedrock Titan Embeddings V2 in controlled batches with live progress
    so you can monitor speed and avoid new-account API throttling.
    """
    print(f"\nConnecting to AWS Bedrock in {BEDROCK_REGION} (amazon.titan-embed-text-v2:0)...")
    embeddings = BedrockEmbeddings(
        model_id="amazon.titan-embed-text-v2:0",
        region_name=BEDROCK_REGION
    )
    
    batch_size = 25
    total = len(chunks)
    vector_db = None

    for i in range(0, total, batch_size):
        batch = chunks[i : i + batch_size]
        if vector_db is None:
            vector_db = FAISS.from_documents(batch, embeddings)
        else:
            vector_db.add_documents(batch)
            
        done = min(i + batch_size, total)
        print(f"  -> Embedded {done}/{total} chunks...")
        time.sleep(0.2)  # Gentle pause to prevent AWS Bedrock rate-limit retries
    
    vector_db.save_local(save_path)
    print(f"\nDone! Vector database saved to '{save_path}/'.")

if __name__ == "__main__":
    raw_docs = load_and_parse_pdfs(DATA_FOLDER)
    chunks = split_documents(raw_docs)
    create_vector_store(chunks, VECTOR_DB_PATH)