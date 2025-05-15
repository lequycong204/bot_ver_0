from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
import google.generativeai as genai

API_KEY = "AIzaSyDFrt-C2FM7Ob4D8ARkXl2vP3s-8maocGs" #gemini
vectodb_path = "faiss_index"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def load_llm():
    genai.configure(api_key=API_KEY)
    llm = genai.GenerativeModel("gemini-2.0-flash")
    return llm

def load_db(embedding_model):
    vector_db = FAISS.load_local(vectodb_path, embedding_model, allow_dangerous_deserialization=True)
    return vector_db


vectorstore = load_db(embedding_model)
llm = load_llm()


query = "Em chào các anh chị ạ! Em là 2k4 có nguyện vọng vào khoa Toán Cơ Tin học của trường ạ"

results = vectorstore.similarity_search(query, k=2)

reference = results[0].metadata["answer"]

# PromptTemplate
prompt = PromptTemplate.from_template(
    """Bạn là trợ lí có kiến thức đầy đủ về vấn đề tuyển sinh trường Đại học Khoa học Tự nhiên
    Dựa vào các tài liệu sau: {context} \n"
    Trả lời đầy đủ, chính xác cho tôi câu hỏi: {query}\n"
    Nếu trong các tài liệu không có thông tin nào liên quan đến câu hỏi của tôi, hãy trả lời "Dựa vào kiến thức của tôi, ..."\n"
    "Output:"""
)

final_prompt = prompt.format(context=reference, query=query)

response = llm.generate_content(final_prompt)

# answer
print(response.text)

