"""
llm.py
统一初始化 DeepSeek 模型实例。
"""
import os
from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek

# 1. 读取 .env 配置
load_dotenv(override=True)

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL")

# 2. 模型初始化
llm_deepseek = ChatDeepSeek(
    model="deepseek-flash",
    api_key=DEEPSEEK_API_KEY,
    api_base=DEEPSEEK_BASE_URL,
    temperature=0,           # 财务场景建议用 0，减少随机性
)

def get_llm():
    """供其他节点统一调用的入口"""
    return llm_deepseek