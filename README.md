---
title: RAG
emoji: 👁
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# RAG Model for Solana SuperTeam Chatbot

Mô hình RAG (Retrieval Augmented Generation) cho Solana SuperTeam Chatbot sử dụng Gemini-1.5-flash và Pinecone để lưu trữ và truy xuất dữ liệu liên quan đến Solana SuperTeam.

## Cấu trúc

```
.
├── NLP_model/
│   └── chatbot.py       # Chứa logic của mô hình RAG
├── app.py               # FastAPI server
├── requirements.txt     # Thư viện cần thiết
├── Dockerfile          # Docker configuration
├── docker-compose.yml  # Docker Compose configuration
├── .dockerignore      # Docker ignore file
└── .env               # Biến môi trường (API keys)
```

## Cài đặt

### Cài đặt trực tiếp

1. Cài đặt các thư viện:

```bash
pip install -r requirements.txt
```

2. Cấu hình API keys trong file `.env`:

```
GOOGLE_API_KEY=your_google_api_key
PINECONE_API_KEY=your_pinecone_api_key
```

3. Chạy API:

```bash
python app.py
```

### Cài đặt bằng Docker

1. Đảm bảo đã cài đặt Docker và Docker Compose

2. Cấu hình API keys trong file `.env`:

```
GOOGLE_API_KEY=your_google_api_key
PINECONE_API_KEY=your_pinecone_api_key
```

3. Build và chạy container:

```bash
docker-compose up --build
```

API sẽ chạy trên `http://localhost:8000`.

## API Endpoints

### POST /chat

Gửi câu hỏi đến mô hình RAG:

```json
{
  "query": "Câu hỏi của người dùng",
  "user_id": "id_của_người_dùng"
}
```

Response:

```json
{
  "response": "Câu trả lời từ mô hình RAG"
}
```

### GET /health

Kiểm tra trạng thái của API:

```json
{
  "status": "healthy"
}
```

## Lưu ý

- Mô hình sử dụng Pinecone index "testbot768" để lưu trữ và truy xuất thông tin.
- Nếu Pinecone không khả dụng, mô hình sẽ cố gắng sử dụng FAISS local index nếu có.
- Mô hình lưu lịch sử trò chuyện cho mỗi người dùng để cung cấp phản hồi phù hợp với ngữ cảnh.
- Khi sử dụng Docker, các biến môi trường sẽ được tự động load từ file .env.
- Container sẽ tự động restart nếu gặp lỗi hoặc server được khởi động lại. 