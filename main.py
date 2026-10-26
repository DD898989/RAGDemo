"""
Google Gemini RAG (Retrieval-Augmented Generation) 知識庫檢索與問答系統
"""

import os
import sys
import warnings

# 忽略第三方套件過渡期警告以保持終端輸出整潔
warnings.filterwarnings("ignore")

from dotenv import load_dotenv

# 載入環境變數（從 .env 檔案）
load_dotenv()

# 讀取 Google API Key（完全透過 .env 檔案或系統環境變數提供）
api_key = os.environ.get("GOOGLE_API_KEY")

if not api_key:
    print("❌ 錯誤：未在 .env 檔案中找到 GOOGLE_API_KEY！")
    print("請在專案目錄下的 .env 檔案中設定：")
    print("GOOGLE_API_KEY=您的_Google_API_Key\n")
    sys.exit(1)

from langchain_classic.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


def main():
    print("=" * 65)
    print("🚀 歡迎使用 RAG 知識庫問答系統 (Google Gemini + FAISS)")
    print("=" * 65)

    # -------------------------------------------------------------------------
    # Step 1: 準備知識庫文檔
    # -------------------------------------------------------------------------
    print("\n【Step 1】準備知識庫文檔")
    print("-" * 50)
    documents = [
        "Amazon S3 是一個物件儲存服務，提供產業領先的可擴展性、資料可用性、安全性及效能。",
        "Amazon EC2 提供安全且可調整大小的運算容量，讓您可以在雲端中執行應用程式。",
        "Amazon RDS 讓您能夠在雲端輕鬆設定、操作和擴展關聯式資料庫。",
        "AWS Lambda 是一項無伺服器運算服務，可讓您執行程式碼而無需佈建或管理伺服器。",
        "Amazon Bedrock 是一項全受管服務，透過單一 API 提供來自領先 AI 公司的高效能基礎模型。"
    ]

    for idx, doc in enumerate(documents, 1):
        print(f"  [{idx}] {doc}")

    # -------------------------------------------------------------------------
    # Step 2: 將文檔轉換成向量並存入 FAISS
    # -------------------------------------------------------------------------
    print("\n【Step 2】將文檔轉換成向量 (Embedding) 並建立本地向量資料庫")
    print("-" * 50)
    print("正在使用 Google Gemini Embedding 模型生成向量並寫入 FAISS...")

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=api_key
    )

    # 文本分割器（將文檔切分成適當大小的區塊 Chunks）
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    docs = text_splitter.create_documents(documents)

    # 創建 FAISS 向量資料庫
    vectorstore = FAISS.from_documents(
        documents=docs,
        embedding=embeddings
    )
    print("✅ FAISS 向量資料庫建立完成！")

    # -------------------------------------------------------------------------
    # Step 3: 實作檢索功能 (Similarity Search)
    # -------------------------------------------------------------------------
    print("\n【Step 3】實作並測試相似度檢索功能")
    print("-" * 50)

    def retrieve_relevant_docs(query, k=3):
        """根據查詢檢索最相關的文檔"""
        return vectorstore.similarity_search(query, k=k)

    test_query = "什麼是無伺服器運算？"
    print(f"🔍 檢索測試問題：「{test_query}」")
    relevant_docs = retrieve_relevant_docs(test_query, k=2)

    print("檢索到的相關文檔：")
    for i, doc in enumerate(relevant_docs, 1):
        print(f"  {i}. {doc.page_content}")

    # -------------------------------------------------------------------------
    # Step 4: 整合 LLM 生成答案 (自訂 Prompt 模式)
    # -------------------------------------------------------------------------
    print("\n【Step 4】整合 Google Gemini 生成答案 (自訂 Prompt + Context)")
    print("-" * 50)

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.2,
        google_api_key=api_key
    )

    def generate_rag_response(query):
        # 1. 檢索相關文檔
        docs = retrieve_relevant_docs(query, k=3)
        # 2. 構建上下文
        context = "\n\n".join([d.page_content for d in docs])
        # 3. 構建 Prompt
        prompt = f"""請根據以下參考資料回答問題。如果參考資料中沒有相關資訊，請誠實地說不知道。

參考資料：
{context}

問題：{query}

答案："""
        response = llm.invoke(prompt)
        return response.content, docs

    query4 = "AWS 有哪些無伺服器服務？"
    print(f"❓ 問題：{query4}")
    answer4, sources4 = generate_rag_response(query4)
    print(f"💡 答案：\n{answer4}\n")
    print("📚 參考來源：")
    for i, doc in enumerate(sources4, 1):
        print(f"  [{i}] {doc.page_content}")

    # -------------------------------------------------------------------------
    # Step 5: 使用 LangChain QA Chain 簡化流程
    # -------------------------------------------------------------------------
    print("\n【Step 5】使用 LangChain RetrievalQA 簡化流程")
    print("-" * 50)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )

    query5 = "Amazon S3 的主要特點是什麼？"
    print(f"❓ 問題：{query5}")
    result = qa_chain.invoke({"query": query5})
    print(f"💡 答案：\n{result['result']}\n")
    print("📚 參考文檔：")
    for doc in result["source_documents"]:
        print(f"  - {doc.page_content}")

    print("\n" + "=" * 65)
    print("🎉 恭喜！RAG 五大核心步驟展示完成！")
    print("=" * 65)

    # -------------------------------------------------------------------------
    # 互動式問答體驗
    # -------------------------------------------------------------------------
    print("\n💬 進入即時問答模式（直接輸入問題按 Enter；輸入 'exit' 或 'q' 結束）：")
    while True:
        user_input = input("\n請輸入您的問題 > ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "q"):
            print("👋 感謝使用，再見！")
            break

        res = qa_chain.invoke({"query": user_input})
        ans = res["result"]
        sources = res["source_documents"]

        print(f"\n💡 回答：\n{ans}")
        print("\n📚 引用文檔：")
        for doc in sources:
            print(f"  - {doc.page_content}")


if __name__ == "__main__":
    main()
