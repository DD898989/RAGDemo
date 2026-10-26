# RAG (Retrieval-Augmented Generation) 知識庫問答系統

今天，我們要介紹一個強大的技術：**RAG (Retrieval-Augmented Generation，檢索增強生成)**，它能讓大型語言模型 (LLM) 基於您的專有知識庫來回答問題。

---

## 📖 簡單介紹 RAG

RAG 是一種結合「**檢索**」和「**生成**」的技術架構：

- **檢索 (Retrieval)**：從專有知識庫中找出與使用者問題最相關的文檔片段。
- **增強 (Augmented)**：將檢索到的相關內容作為上下文 (Context) 注入到 Prompt 中。
- **生成 (Generation)**：讓 LLM 基於這些檢索到的上下文生成精確答案。

### 🌟 RAG 的核心好處

1. **即時更新**：不需要耗時費力地重新訓練 (Fine-tuning) 模型，只要更新文檔或向量庫就能使用最新資料。
2. **可追溯性**：答案可以精確附上引用來源，清楚知道答案來自哪些文檔。
3. **降低幻覺**：基於實際提供的文檔回答，大幅減少模型虛構或編造內容的情況。
4. **成本效益**：相較於模型微調 (Fine-tuning)，RAG 建置成本極低且維護彈性極高。

---

## 🏗️ RAG 架構圖

```text
使用者問題
    ↓
問題向量化 (Embedding)
    ↓
向量資料庫搜尋 (Vector Search / FAISS)
    ↓
檢索相關文檔 (Top-k Chunks)
    ↓
構建 Prompt (問題 + 參考資料上下文)
    ↓
LLM 生成答案 (Gemini / Claude)
    ↓
回傳給使用者 (附帶參考文檔來源)
```

---

### 🔑 環境變數設定

在專案目錄下的 `.env` 檔案中填入您的 Google API Key：

```env
GOOGLE_API_KEY=您的_Google_API_Key
```

> 💡 專案內已包含 `.env.example` 供參考。程式不會將金鑰寫死在任何程式碼內，全數透過 `python-dotenv` 安全載入。

---

## 🚀 快速開始

### 方式一：一鍵執行（推薦）

直接雙擊專案目錄下的 **`RUN.bat`** 即可：
- 自動偵測 `uv` 或 `python`。
- 自動建立虛擬環境與安裝相依套件。
- 自動由 `.env` 載入 `GOOGLE_API_KEY`。
- 依序展示 Step 1 ~ Step 4，最後進入**即時互動問答模式**！

### 方式二：使用命令列執行

```bash
# 透過 uv 執行
uv run python main.py

# 或使用原生 Python
python -m venv .venv
.venv\Scripts\activate
pip install langchain langchain-community langchain-google-genai langchain-text-splitters faiss-cpu python-dotenv
python main.py
```

---

## 🛠️ 安裝相依套件

本專案支援使用 `uv` 快速管理套件：

```bash
uv add langchain langchain-community langchain-google-genai langchain-text-splitters faiss-cpu python-dotenv
```

---

## 📚 實作步驟詳解

本專案使用 **Google Gemini** 與 **FAISS** 實作核心的 5 個步驟：

### Step 1: 準備知識庫文檔

```python
documents = [
    "Amazon S3 是一個物件儲存服務，提供產業領先的可擴展性、資料可用性、安全性及效能。",
    "Amazon EC2 提供安全且可調整大小的運算容量，讓您可以在雲端中執行應用程式。",
    "Amazon RDS 讓您能夠在雲端輕鬆設定、操作和擴展關聯式資料庫。",
    "AWS Lambda 是一項無伺服器運算服務，可讓您執行程式碼而無需佈建或管理伺服器。",
    "Amazon Bedrock 是一項全受管服務，透過單一 API 提供來自領先 AI 公司的高效能基礎模型。"
]
```

### Step 2: 將文檔轉換成向量並建立向量資料庫 (FAISS)

透過 Embedding 模型將文字轉換成多維語意向量，並存入本地高效向量資料庫 FAISS：

```python
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# 初始化 Embedding 模型 (Google Gemini Embedding)
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)

# 文本切塊器 (將文檔切分成適當大小的區塊)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
docs = text_splitter.create_documents(documents)

# 建立 FAISS 本地向量資料庫
vectorstore = FAISS.from_documents(
    documents=docs,
    embedding=embeddings
)
print("✅ 向量資料庫建立完成！")
```

### Step 3: 實作相似度檢索功能

透過餘弦相似度或向量距離，找出與使用者問題語意最接近的前 $k$ 個文檔片段：

```python
def retrieve_relevant_docs(query, k=3):
    """根據查詢檢索最相關的文檔"""
    return vectorstore.similarity_search(query, k=k)

# 測試檢索
query = "什麼是無伺服器運算？"
relevant_docs = retrieve_relevant_docs(query, k=2)

print(f"問題：{query}\n")
for i, doc in enumerate(relevant_docs, 1):
    print(f"{i}. {doc.page_content}")
```

### Step 4: 整合 LLM 生成答案 (自訂 Prompt 模式)

將檢索到的文檔內容串接為參考資料，組裝 Prompt 後送交模型回答：

```python
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)

def generate_rag_response(query):
    # 1. 檢索相關文檔
    relevant_docs = retrieve_relevant_docs(query, k=3)
    
    # 2. 構建上下文
    context = "\n\n".join([doc.page_content for doc in relevant_docs])
    
    # 3. 構建 Prompt
    prompt = f"""請根據以下參考資料回答問題。如果參考資料中沒有相關資訊，請誠實地說不知道。

參考資料：
{context}

問題：{query}

答案："""
    
    response = llm.invoke(prompt)
    return response.content, relevant_docs

query = "AWS 有哪些無伺服器服務？"
answer, sources = generate_rag_response(query)
print(f"問題：{query}\n答案：{answer}")
```

### Step 5: 使用 LangChain 簡化流程

使用 LangChain 封裝好的 `RetrievalQA` Chain，大幅簡化檢索與生成的串接程式碼：

```python
from langchain_classic.chains import RetrievalQA

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True
)

# 執行查詢
result = qa_chain.invoke({"query": "Amazon S3 的主要特點是什麼？"})

print(f"答案：{result['result']}\n")
print("參考文檔：")
for doc in result["source_documents"]:
    print(f"- {doc.page_content}")
```

---

## 📁 專案檔案清單

- **`RUN.bat`**：Windows 一鍵執行批次檔（支援 UTF-8、自動偵測環境與啟動）
- **`main.py`**：Google Gemini + FAISS 完整 RAG 實作與即時互動終端
- **`pyproject.toml`**：uv 專案設定與 Python 相依性清單
- **`.env.example`**：環境變數設定範例檔
- **`.gitignore`**：Git 忽略規則，自動過濾虛擬環境與金鑰檔案
