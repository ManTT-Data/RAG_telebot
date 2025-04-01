import os
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
# from langchain_ollama import OllamaLLM
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import threading
from datetime import datetime
from langchain.schema import HumanMessage, AIMessage
from langchain_google_genai import GoogleGenerativeAIEmbeddings
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

#lấy model chatbot
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-8b-latest",
                             temperature=0.8)
# model = OllamaLLM(model="llama2")
# print("Llama2 đã được tải thành công!")

#lấy model embedding
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")


# Biến lưu history cho từng user (dạng chuỗi)
user_histories = {}
history_lock = threading.Lock()

# Create a prompt template with conversation history
prompt = PromptTemplate(
    template = """Goal:
You are a professional tour guide assistant that assists users in finding information about places in Da Nang, Vietnam.
You can provide details on restaurants, cafes, hotels, attractions, and other local venues. You have to chat with users, who are Da Nang tourists. 

Return Format:
Respond in friendly, natural, and concise English like a real tour guide. 

Warning:
Let's support users like a real tour guide, not a bot. The information in context is your own knowledge.
Your knowledge is provided in the Context. All of information in Context is about Da Nang, Vietnam.
You just care about current time that user mention when user ask about Solana event.
If you do not have enough information to answer user's question, reply with "I don't know. I don't have information about that".

Context:
{context}

Conversation History:
{chat_history}

User chat:
{question}

Your chat:
""",
    input_variables = ["context", "question", "chat_history"],
)

def get_history(user_id):
    """Get conversation history for a specific user"""
    with history_lock:
        return user_histories.get(user_id, "")

def update_history(user_id, new_entry):
    """Update conversation history for a user.
    new_entry should be a string containing the new conversation information, e.g.:
      "User: {question}\nBot: {answer}\n"
    """
    with history_lock:
        current_history = user_histories.get(user_id, "")
        # Store only the last 30 interactions by keeping the 60 most recent lines
        # (assuming 2 lines per interaction: 1 for user, 1 for bot)
        history_lines = current_history.split('\n')
        if len(history_lines) > 60:
            history_lines = history_lines[-60:]
            current_history = '\n'.join(history_lines)
        
        updated_history = current_history + new_entry + "\n"
        user_histories[user_id] = updated_history

def string_to_message_history(history_str):
    """Convert string-based history to LangChain message history format"""
    if not history_str.strip():
        return []
    
    messages = []
    lines = history_str.strip().split('\n')
    i = 0
    
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("User:"):
            user_message = line[5:].strip()  # Get the user message without "User:"
            messages.append(HumanMessage(content=user_message))
            
            # Look for a Bot response (should be the next line)
            if i + 1 < len(lines) and lines[i + 1].strip().startswith("Bot:"):
                bot_response = lines[i + 1][4:].strip()  # Get bot response without "Bot:"
                messages.append(AIMessage(content=bot_response))
                i += 2  # Skip the bot line too
            else:
                i += 1
        else:
            i += 1  # Skip any unexpected format lines
            
    return messages

def get_chain():
    """Get the retrieval chain with Pinecone vector store"""
    try:
        pc = Pinecone(
            api_key=os.environ["PINECONE_API_KEY"]
        )
        
        # Get the vector store from the existing index
        vectorstore = PineconeVectorStore.from_existing_index(
            index_name="testbot768",
            embedding=embeddings,
            text_key="text"
        )
        
        retrieve = vectorstore.as_retriever(search_kwargs={"k": 3})
            
        return retrieve
    except Exception as e:
        print(f"Error getting vector store: {e}")
        return None

def chat(request, user_id="default_user"):
    """Process a chat request from a specific user"""
    try:
        # Get retrieval chain
        retriever = get_chain()
        if not retriever:
            return "Error: Could not initialize retriever"
        
        # Get current conversation history as string
        conversation_history_str = get_history(user_id)
        
        # Convert string history to LangChain message format
        message_history = string_to_message_history(conversation_history_str)
        
        # Get current time
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Add timestamp to question
        question_with_time = f"{request}\n(Current time: {current_time})"
        # print("User question:", question_with_time)
        
        # Create a ConversationalRetrievalChain
        # Get relevant documents from retriever
        retrieved_docs = retriever.get_relevant_documents(question_with_time)
        print("Retrieved documents page content:", [doc.page_content for doc in retrieved_docs])
        
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=model,
            retriever=retriever,
            combine_docs_chain_kwargs={"prompt": prompt}
        )
        
        # Call the chain with question and converted message history
        response = conversation_chain({"question": question_with_time, "chat_history": message_history})
        answer = str(response['answer'])
        
        # Update conversation history string
        new_entry = f"User: {question_with_time}\nBot: {answer}"
        update_history(user_id, new_entry)
        print(get_history(user_id))
        
        print(answer)
        return answer
    except Exception as e:
        print(f"Error in chat: {e}")
        return f"I encountered an error: {str(e)}"

def clear_memory(user_id="default_user"):
    """Clear the conversation history for a specific user"""
    with history_lock:
        if user_id in user_histories:
            del user_histories[user_id]
            return f"Conversation history cleared for user {user_id}"
        return f"No conversation history found for user {user_id}"