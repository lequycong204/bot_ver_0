import streamlit as st
from app import load_llm, load_db, embedding_model, PromptTemplate
import json
from datetime import datetime
import os
import pickle

class HUSChatbot:
    def __init__(self):
        # Tạo thư mục cache nếu chưa tồn tại
        self.CACHE_DIR = "cache"
        if not os.path.exists(self.CACHE_DIR):
            os.makedirs(self.CACHE_DIR)
            
        self.CACHE_FILE = os.path.join(self.CACHE_DIR, "current_chat.pkl")
        self.vectorstore = None
        self.llm = None
        self.initialize_models()
        self.initialize_session_state()
        self.setup_page_config()
        self.setup_css()

    def initialize_models(self):
        """Khởi tạo các model cần thiết"""
        self.vectorstore = load_db(embedding_model)
        self.llm = load_llm()

    def initialize_session_state(self):
        """Khởi tạo trạng thái session"""
        if "messages" not in st.session_state:
            if os.path.exists(self.CACHE_FILE):
                try:
                    with open(self.CACHE_FILE, 'rb') as f:
                        st.session_state.messages = pickle.load(f)
                except:
                    st.session_state.messages = []
            else:
                st.session_state.messages = []

    def setup_page_config(self):
        """Cấu hình trang Streamlit"""
        st.set_page_config(
            page_title="HUS Chatbot",
            page_icon="🎓",
            layout="wide"
        )

    def setup_css(self):
        """Thiết lập CSS cho giao diện"""
        st.markdown("""
        <style>
        .stTextInput>div>div>input {
            border-radius: 20px;
            position: fixed;
            bottom: 3rem;
            left: 20%;
            width: 60%;
            z-index: 100;
            background-color: #1E1E1E;
            border: 1px solid #4B4B4B;
            padding: 1rem;
            color: white;
        }

        .stTextInput>div>div>input:focus {
            border-color: #6B6B6B;
            box-shadow: 0 0 0 1px #6B6B6B;
        }

        .chat-message {
            padding: 1.5rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
            display: flex;
            flex-direction: column;
        }

        .chat-message.user {
            background-color: #2b313e;
            margin-left: 20%;
            margin-right: 1rem;
        }

        .chat-message.assistant {
            background-color: #475063;
            margin-right: 20%;
            margin-left: 1rem;
        }

        .chat-message .avatar {
            width: 20px;
            height: 20px;
            border-radius: 50%;
            margin-right: 0.5rem;
        }

        .sidebar-content {
            padding: 1rem;
        }

        .main .block-container {
            padding-bottom: 100px;
        }

        .chat-container {
            height: calc(100vh - 200px);
            overflow-y: auto;
            padding: 1rem;
            margin-bottom: 80px;
        }
        </style>
        """, unsafe_allow_html=True)

    def save_to_cache(self):
        """Lưu chat hiện tại vào cache"""
        with open(self.CACHE_FILE, 'wb') as f:
            pickle.dump(st.session_state.messages, f)

    def clear_cache(self):
        """Xóa cache"""
        if os.path.exists(self.CACHE_FILE):
            os.remove(self.CACHE_FILE)
        st.session_state.messages = []

    def generate_response(self, prompt):
        """Sinh câu trả lời cho câu hỏi"""
        # Get relevant documents
        results = self.vectorstore.similarity_search(prompt, k=2)
        reference = results[0].metadata["answer"]

        # Create prompt template
        prompt_template = PromptTemplate.from_template(
            """Bạn là trợ lí có kiến thức đầy đủ về vấn đề tuyển sinh trường Đại học Khoa học Tự nhiên
            Dựa vào các thông tin sau: {context} \n"
            Trả lời đầy đủ, chính xác cho tôi câu hỏi: {query}\n"
            "Output:"
            Nếu bạn còn thắc mắc gì, tôi sẽ giải đáp thêm cho bạn"""
        )

        final_prompt = prompt_template.format(context=reference, query=prompt)
        
        # Generate response
        response = self.llm.generate_content(final_prompt)
        return response.text

    def display_sidebar(self):
        """Hiển thị thanh bên"""
        with st.sidebar:
            st.title("🎓 HUS Chatbot")
            st.markdown("---")
            
            if st.button("🗑️ Bắt đầu cuộc hội thoại mới"):
                self.clear_cache()
                st.success("Đã bắt đầu cuộc hội thoại mới!")
                st.rerun()

    def display_chat_messages(self):
        """Hiển thị tin nhắn chat"""
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

    def process_user_input(self, prompt):
        """Xử lý input từ người dùng"""
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate and display response
        with st.chat_message("assistant"):
            with st.spinner("Đang suy nghĩ..."):
                response = self.generate_response(prompt)
                # Clear the spinner and display response
                st.empty()
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
                
                # Auto-save current chat to cache
                self.save_to_cache()

    def run(self):
        """Chạy chatbot"""
        # Display sidebar
        self.display_sidebar()

        # Main chat interface
        st.title("💬 Chat với HUS Bot")
        st.markdown("Chào mừng bạn đến với chatbot tư vấn tuyển sinh trường Đại học Khoa học Tự nhiên!")

        # Display chat messages
        self.display_chat_messages()

        # Chat input
        if prompt := st.chat_input("Nhập câu hỏi của bạn..."):
            self.process_user_input(prompt)

def main():
    chatbot = HUSChatbot()
    chatbot.run()

if __name__ == "__main__":
    main() 