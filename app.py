from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from NLP_model import chatbot
import uvicorn
import asyncio
import time
import logging
from contextlib import asynccontextmanager
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Chuẩn bị RAG model tại lúc khởi động
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Khởi tạo retriever sẵn khi server bắt đầu
    logger.info("Initializing RAG model retriever...")
    # Sử dụng asyncio.to_thread để không block event loop
    await asyncio.to_thread(chatbot.get_chain)
    logger.info("RAG model retriever initialized successfully")
    yield
    # Dọn dẹp khi shutdown
    logger.info("Shutting down RAG model...")

app = FastAPI(
    title="Solana SuperTeam RAG API", 
    description="API cho mô hình RAG của Solana SuperTeam",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request counter để theo dõi số lượng request đang xử lý
active_requests = 0
max_concurrent_requests = 5  # Giới hạn số request xử lý đồng thời
request_lock = asyncio.Lock()

class ChatRequest(BaseModel):
    query: str
    user_id: str = "default_user"

class ChatResponse(BaseModel):
    response: str
    processing_time: float = None

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Middleware để đo thời gian xử lý và kiểm soát số lượng request"""
    global active_requests
    
    # Kiểm tra và tăng số request đang xử lý
    async with request_lock:
        # Nếu đã đạt giới hạn, từ chối request mới
        if active_requests >= max_concurrent_requests and request.url.path == "/chat":
            return JSONResponse(
                status_code=429,
                content={"detail": "Too many requests. Please try again later."}
            )
        active_requests += 1
    
    try:
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        
        # Thêm thời gian xử lý vào header
        response.headers["X-Process-Time"] = str(process_time)
        logger.info(f"Request processed in {process_time:.2f} seconds: {request.url.path}")
        return response
    finally:
        # Giảm counter khi xử lý xong
        async with request_lock:
            active_requests -= 1

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Xử lý yêu cầu chat từ người dùng
    """
    start_time = time.time()
    try:
        # Gọi hàm chat với thông tin được cung cấp
        response = await asyncio.to_thread(chatbot.chat, request.query, int(request.user_id))
        process_time = time.time() - start_time
        return ChatResponse(
            response=response,
            processing_time=process_time
        )
    except Exception as e:
        logger.error(f"Error processing chat request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """
    Kiểm tra trạng thái của API
    """
    # Kiểm tra xem retriever đã được khởi tạo chưa
    retriever = chatbot.get_chain()
    if retriever:
        status = "healthy"
    else:
        status = "degraded"
    
    return {
        "status": status,
        "active_requests": active_requests,
        "cache_size": len(chatbot.response_cache)
    }

@app.post("/clear-memory/{user_id}")
async def clear_user_memory(user_id: str):
    """
    Xóa lịch sử trò chuyện của một người dùng
    """
    try:
        result = await asyncio.to_thread(chatbot.clear_memory, user_id)
        return {"status": "success", "message": result}
    except Exception as e:
        logger.error(f"Error clearing memory for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import nest_asyncio
    nest_asyncio.apply()
    uvicorn.run(app, host="0.0.0.0", port=7860) 