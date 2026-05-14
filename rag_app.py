import os
os.environ["CHROMA_TELEMETRY_IMPL"] = "none"

import tempfile
from dotenv import load_dotenv
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI

load_dotenv()

# 初始化 DeepSeek
deepseek_client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1"
)

# 初始化 Chroma（使用本地 embedding）
@st.cache_resource
@st.cache_resource
def init_chroma():
    # 改用内存模式（临时存储）
    chroma_client = chromadb.EphemeralClient()
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="BAAI/bge-small-zh-v1.5"
    )
    return chroma_client, embed_fn

chroma_client, embed_fn = init_chroma()

st.set_page_config(page_title="RAG 問答助理", layout="wide")
st.title("📄 你的文件問答助理")
st.markdown("上傳 PDF，然後問問題。系統會從文件中找答案（可選精確或總結模式）。")

# 側邊欄：模式選擇
mode = st.sidebar.radio("選擇模式", ["精確模式 (RAG)", "總結模式 (全文)"])
uploaded_file = st.sidebar.file_uploader("上傳 PDF 文件", type=["pdf"])

if uploaded_file is not None:
    # 儲存上傳的檔案到暫存
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    # 讀取 PDF 內容
    loader = PyPDFLoader(tmp_path)
    docs = loader.load()
    full_text = "\n".join([doc.page_content for doc in docs])

    # 對於總結模式：直接顯示輸入框，不使用 RAG
    if mode == "總結模式 (全文)":
        st.info("總結模式：將整份文件送到 AI，適合問『主要內容』、『研究目標』等全局問題。")
        user_question = st.text_input("你的問題：")
        if st.button("送出") and user_question:
            with st.spinner("AI 閱讀整份文件中..."):
                # 限制前 30000 字（避免過長）
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

    else:  # 精確模式 (RAG)
        st.info("精確模式：先從文件中檢索相關段落（前 5 段），再由 AI 根據這些段落回答。適合具體事實問題。")
        # 建立一個以檔案名稱命名的 collection
        collection_name = uploaded_file.name.replace(".", "_")
        collection = chroma_client.get_or_create_collection(
            name=collection_name,
            embedding_function=embed_fn
        )
        # 如果 collection 是空的，才需要索引（避免重複）
        if collection.count() == 0:
            with st.spinner("正在分割並索引文件（第一次需要，之後會加快）..."):
                splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                chunks = splitter.split_documents(docs)
                for i, chunk in enumerate(chunks):
                    collection.add(
                        documents=[chunk.page_content],
                        metadatas=[{"source": uploaded_file.name, "chunk_id": i}],
                        ids=[f"{uploaded_file.name}_{i}"]
                    )
                st.success(f"已索引 {len(chunks)} 個區塊")
        else:
            st.info(f"文件中已有 {collection.count()} 個區塊，可直接查詢。")

        user_question = st.text_input("你的問題：")
        top_k = st.slider("檢索相關段落數量", 1, 10, 5)
        if st.button("送出") and user_question:
            with st.spinner("檢索中..."):
                results = collection.query(query_texts=[user_question], n_results=top_k)
                retrieved_chunks = results['documents'][0]
                if not retrieved_chunks:
                    st.warning("找不到相關段落，請換個問法。")
                else:
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