import os
os.environ["CHROMA_TELEMETRY_IMPL"] = "none"  # 保留无害

import tempfile
from dotenv import load_dotenv
import streamlit as st
import numpy as np
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from openai import OpenAI

load_dotenv()

deepseek_client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1"
)

@st.cache_resource
def load_embedding_model():
    # 使用轻量英文模型（下载快，但中文效果稍弱；也可换 BAAI/bge-small-zh-v1.5，但稍大）
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_embedding_model()

st.set_page_config(page_title="RAG 問答助理", layout="wide")
st.title("📄 你的文件問答助理")
st.markdown("上傳 PDF，然後問問題。系統會從文件中找答案（可選精確或總結模式）。")

mode = st.sidebar.radio("選擇模式", ["精確模式 (RAG)", "總結模式 (全文)"])
uploaded_file = st.sidebar.file_uploader("上傳 PDF 文件", type=["pdf"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    loader = PyPDFLoader(tmp_path)
    docs = loader.load()
    full_text = "\n".join([doc.page_content for doc in docs])

    if mode == "總結模式 (全文)":
        st.info("總結模式：將整份文件送到 AI，適合問『主要內容』、『研究目標』等全局問題。")
        user_question = st.text_input("你的問題：")
        if st.button("送出") and user_question:
            with st.spinner("AI 閱讀整份文件中..."):
                if len(full_text) > 30000:
                    display_text = full_text[:30000]
                else:
                    display_text = full_text
                prompt = f"請根據以下文件回答問題。如果答案在文件中，請給出；如果沒有，請說『文件中未提及』。\n\n文件：\n{display_text}\n\n問題：{user_question}\n\n回答："
                response = deepseek_client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=800
                )
                st.success("✅ 回答：")
                st.write(response.choices[0].message.content)

    else:  # 精確模式 (RAG) – 无 chromadb，使用 numpy 检索
        st.info("精確模式：先從文件中檢索相關段落，再由 AI 根據這些段落回答。適合具體事實問題。")
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(docs)
        chunk_texts = [chunk.page_content for chunk in chunks]
        st.write(f"文件已分割為 {len(chunk_texts)} 個段落")

        # 计算所有 chunk 的 embedding（缓存，每个会话只计算一次）
        if "chunk_embeddings" not in st.session_state:
            with st.spinner("計算段落向量中..."):
                st.session_state.chunk_embeddings = model.encode(chunk_texts, normalize_embeddings=True)
                st.session_state.chunk_texts = chunk_texts
        else:
            chunk_texts = st.session_state.chunk_texts

        user_question = st.text_input("你的問題：")
        top_k = st.slider("檢索相關段落數量", 1, 10, 5)

        if st.button("送出") and user_question:
            with st.spinner("檢索中..."):
                q_emb = model.encode([user_question], normalize_embeddings=True)
                similarities = np.dot(st.session_state.chunk_embeddings, q_emb.T).flatten()
                top_indices = np.argsort(similarities)[-top_k:][::-1]
                retrieved_chunks = [st.session_state.chunk_texts[i] for i in top_indices]

                context = "\n\n---\n\n".join(retrieved_chunks)
                prompt = f"你是一個基於文件的問答助手。請只用以下提供的資料來回答問題。如果資料不夠，請說「根據現有文件無法回答這個問題」。\n\n資料：\n{context}\n\n問題：{user_question}\n\n回答："
                response = deepseek_client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=500
                )
                st.success("✅ 回答：")
                st.write(response.choices[0].message.content)
                with st.expander("查看檢索到的相關段落"):
                    for i, chunk in enumerate(retrieved_chunks):
                        st.markdown(f"**段落 {i+1}**")
                        st.write(chunk)
                        st.markdown("---")