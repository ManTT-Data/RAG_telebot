import os
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import threading
from datetime import datetime
import time
from langchain.schema import HumanMessage, AIMessage
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import functools
import hashlib
import logging
import random
from mongodb import get_chat_history

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configure API keys from environment variables
google_api_key = os.getenv("GOOGLE_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")

if not google_api_key or not pinecone_api_key:
    raise ValueError("Missing required API keys in environment variables")

os.environ["GOOGLE_API_KEY"] = google_api_key
os.environ["PINECONE_API_KEY"] = pinecone_api_key

genai.configure(api_key=google_api_key)

# Lấy model chatbot
try:
    generation_config = {
        "temperature": 0.9,
        "top_p": 1,
        "top_k": 1,
        "max_output_tokens": 2048,
    }

    safety_settings = [
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
        {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
    ]

    model = genai.GenerativeModel(
        model_name='models/gemini-2.0-flash',
        generation_config=generation_config,
        safety_settings=safety_settings
    )
    
    # Test the model with a simple prompt
    test_response = model.generate_content("Hello")
    logger.debug(f"Test response: {test_response.text if hasattr(test_response, 'text') else 'No text attribute'}")
    
except Exception as e:
    logger.error(f"Error initializing or testing Gemini model: {str(e)}")
    raise

# Lấy model embedding
# Print available embedding models
# available_models = GoogleGenerativeAIEmbeddings.list_models()
# embedding_models = [model.name for model in available_models if "embedding" in model.name.lower()]
# logger.info(f"Available embedding models: {embedding_models}")

# Use the embedding model
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
# embeddings = genai.GenerativeModel(model_name="models/embedding-004")

# Cache for responses
response_cache = {}
cache_lock = threading.Lock()
# Maximum cache size và thời gian sống (30 phút)
MAX_CACHE_SIZE = 100
CACHE_TTL = 1800  # 30 phút tính bằng giây

# Create a prompt template with conversation history
prompt = PromptTemplate(
    template = """Goal:
You are a professional tour guide assistant that assists users in finding information about places in Da Nang, Vietnam.
You can provide details on restaurants, cafes, hotels, attractions, and other local venues. 
You have to use core knowledge and conversation history to chat with users, who are Da Nang's tourists. 

Return Format:
Respond in friendly, natural, concise and use only English like a real tour guide. 

Warning:
Let's support users like a real tour guide, not a bot. The information in core knowledge is your own knowledge.
Your knowledge is provided in the Core Knowledge. All of information in Core Knowledge is about Da Nang, Vietnam.
You just care about current time that user mention when user ask about Solana event.
If you do not have enough information to answer user's question, please reply with "I don't know. I don't have information about that".

Core knowledge:
{context}

Conversation History:
{chat_history}

User message:
{question}

Your message:
""",
    input_variables = ["context", "question", "chat_history"],
)

def get_history(user_id):
    """Get conversation history for a specific user from MongoDB"""
    return get_chat_history(user_id)


def get_chain():
    """Get the retrieval chain with Pinecone vector store (singleton pattern)"""
    try:
        start_time = time.time()
        pc = Pinecone(
            api_key=os.environ["PINECONE_API_KEY"]
        )
        
        # Get the vector store from the existing index
        vectorstore = PineconeVectorStore.from_existing_index(
            index_name="testbot768",
            embedding=embeddings,
            text_key="text"
        )
        
        _retriever_instance = vectorstore.as_retriever(search_kwargs={"k": 5})
        logger.info(f"Pinecone retriever initialized in {time.time() - start_time:.2f} seconds")
        return _retriever_instance
    except Exception as e:
        logger.error(f"Error getting vector store from Pinecone: {e}")
        # Fallback to a local vector store or return None
        try:
            # Try to load a local FAISS index if it exists
            start_time = time.time()
            vectorstore = FAISS.load_local("faiss_index", embeddings)
            _retriever_instance = vectorstore.as_retriever(search_kwargs={"k": 3})
            logger.info(f"FAISS retriever initialized in {time.time() - start_time:.2f} seconds")
            return _retriever_instance
        except Exception as faiss_error:
            logger.error(f"Error getting FAISS vector store: {faiss_error}")
            return None

def clean_cache():
    """Clean expired cache entries"""
    with cache_lock:
        current_time = time.time()
        expired_keys = [k for k, v in response_cache.items() if current_time - v['timestamp'] > CACHE_TTL]
        
        for key in expired_keys:
            del response_cache[key]
            
        # Nếu cache vẫn quá lớn, xóa các mục cũ nhất
        if len(response_cache) > MAX_CACHE_SIZE:
            # Sắp xếp theo thời gian và giữ lại MAX_CACHE_SIZE mục mới nhất
            sorted_items = sorted(response_cache.items(), key=lambda x: x[1]['timestamp'])
            items_to_remove = sorted_items[:len(sorted_items) - MAX_CACHE_SIZE]
            
            for key, _ in items_to_remove:
                del response_cache[key]

def generate_cache_key(request, user_id):
    """Generate a unique cache key from the request and user_id"""
    # Tạo một chuỗi kết hợp để hash
    combined = f"{request.strip().lower()}:{user_id}"
    # Tạo MD5 hash
    return hashlib.md5(combined.encode()).hexdigest()

def chat(request, user_id="default_user"):
    """Process a chat request from a specific user"""
    start_time = time.time()
    
    # Định kỳ xóa các mục cache hết hạn
    if random.random() < 0.1:
        clean_cache()
    
    cache_key = generate_cache_key(request, user_id)
    
    with cache_lock:
        if cache_key in response_cache:
            cache_data = response_cache[cache_key]
            if time.time() - cache_data['timestamp'] <= CACHE_TTL:
                logger.info(f"Cache hit for user {user_id}, request: '{request[:30]}...'")
                cache_data['timestamp'] = time.time()
                return cache_data['response']
    try:
        retriever = get_chain()
        if not retriever:
            return "Error: Could not initialize retriever"
        
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Sử dụng invoke thay vì get_relevant_documents
        retrieved_docs = retriever.invoke(request)
        context = "\n".join([doc.page_content for doc in retrieved_docs])
        
        # Sử dụng generate_content thay vì invoke cho model Gemini
        prompt_text = prompt.format(
            context=context,
            question=request,
            chat_history=get_history(user_id)
        )
        print(prompt_text)
        
        response = model.generate_content(prompt_text)
        answer = response.text  # Sử dụng .text thay vì .content
        
        # Lưu vào cache
        with cache_lock:
            response_cache[cache_key] = {
                'response': answer,
                'timestamp': time.time()
            }
        
        logger.info(f"Total processing time: {time.time() - start_time:.2f} seconds")
        return answer
    except Exception as e:
        logger.error(f"Error in chat: {e}")
        return f"I don't know how to answer that right now. Let me forward this to the admin team."
