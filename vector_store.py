import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
#from sentence_transformers import SentenceTransformer

df = pd.read_csv("faq.csv")
vectodb_path = "faiss_index"

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

documents = [
    Document(page_content=row['question'], metadata={"answer": row['answer']})
    for _, row in df.iterrows()
]

# vector store
vectorstore = FAISS.from_documents(documents, embedding_model)
vectorstore.save_local(vectodb_path)