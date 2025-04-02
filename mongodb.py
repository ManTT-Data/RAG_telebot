import os
from pymongo import MongoClient
import logging
from dotenv import load_dotenv

# Load biến môi trường từ .env (nếu có)
load_dotenv()

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Lấy thông tin kết nối MongoDB từ biến môi trường
MONGODB_URI = os.getenv("MONGODB_URI")
MONGODB_DB = os.getenv("MONGODB_DB")
MONGODB_COLLECTION = os.getenv("MONGODB_COLLECTION")

# Kết nối MongoDB sử dụng pymongo
client = MongoClient(MONGODB_URI)
db = client[MONGODB_DB]
collection = db[MONGODB_COLLECTION]


def get_chat_history(user_id: int) -> str:
    """
    Lấy lịch sử chat cho user_id cho trước từ MongoDB và ghép thành chuỗi theo định dạng:
    
    Bot: ...
    User: ...
    Bot: ...
    ...
    
    Giả sử:
      - Các document chứa trường "user_id" để lọc theo user_id.
      - Trường "factor" xác định nguồn tin (nếu factor == "user" thì là tin của User,
        còn lại coi là tin của Bot/RAG).
      - Trường "timestamp" dùng để sắp xếp theo thời gian (nếu có).
    """
    try:
        # Truy vấn tất cả các document có user_id, sắp xếp theo timestamp tăng dần
        # Nếu không có trường timestamp, có thể sort theo _id
        docs = list(collection.find({"user_id": user_id}).sort("timestamp", -1).limit(20))
        docs.reverse()

        if not docs:
            logger.info(f"Không tìm thấy dữ liệu cho user_id: {user_id}")
            return ""
        
        conversation_lines = []
        for doc in docs:
            factor = doc.get("factor", "").lower()
            action = doc.get("action", "").lower()
            message = doc.get("message", "")
            
            if action == "freely asking":
                conversation_lines.append(f"User: {message}")
            elif action == "response":
                conversation_lines.append(f"Bot: {message}")
        
        # Ghép các dòng thành chuỗi, mỗi dòng cách nhau bằng xuống dòng
        logger.info("User ID: " + str(user_id))
        return "\n".join(conversation_lines)
    except Exception as e:
        logger.error(f"Lỗi khi lấy lịch sử chat cho user_id {user_id}: {e}")
        return ""

# if __name__ == '__main__':
#     user_id = int(input("Nhập user_id cần lấy lịch sử chat: ").strip())
#     history = get_chat_history(user_id)
#     if history:
#         print("\nLịch sử trò chuyện:")
#         print(history)
#     else:
#         print(f"Không tìm thấy lịch sử chat cho user_id: {user_id}")
