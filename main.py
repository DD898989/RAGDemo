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

# 檢查 Google API Key（透過 .env 檔案或系統環境變數提供）
if not os.environ.get("GOOGLE_API_KEY"):
    print("❌ 錯誤：未在 .env 檔案中找到 GOOGLE_API_KEY！")
    print("請在專案目錄下的 .env 檔案中設定：")
    print("GOOGLE_API_KEY=您的_Google_API_Key\n")
    sys.exit(1)

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
        "1.根據泡麵發明人日清食品的安藤百福所說，讓泡麵的人等三分鐘，會令人更餓，吃麵食會感到更美味，所以廠商把泡麵時間定在三分鐘。",
        "2.當用遠離門把的那隻手去開汽車門時，上半身會被強迫轉向，同時視線會投射到後方，因此便會注意到後面的車況跟車流，減少撞車危險。",
        "3.牙膏是靠所含清潔成分和牙刷與牙齒的摩擦來清潔的，牙刷不沾水，擠上牙膏慢慢刷，漸漸出現細微的泡沫，牙膏的清潔成分才能被發揮到最大。",
        "4.酒精作用使我們對膀胱的感覺能力變得遲鈍，一般而言，膀胱到達200cc時，就會想去廁所，但剛喝醉酒時，膀胱達到最大極限，還感覺不到尿意。所以，當喝醉酒想去廁所時，尿量一定是平日的兩倍。上廁所的時間也因此變長。",
        "5.用塑膠容器製冰，因為導熱性質差，而且間隔距離比較小，所以熱量不太容易進出，冷凍時間會較長，但是利用金屬容器導熱性非常好，製冰的速度就會變得非常快速。",
        "6.被蚊子咬完時，塗上肥皂就不會癢了。",
        "7.仰頭點眼藥水時微微張嘴，這樣眼睛就不會亂眨了。",
        "8.吃了辣的東西，感覺就要被辣死了，就往嘴裡放上少許鹽，含一下後吐掉，漱個口就不辣了。",
        "9.如果你上廁所時看書，會記得特別牢。",
        "10.牙刷應該放置距離馬桶至少2公尺遠的地方，以避免沖馬桶時產生並漂浮於空氣中的各種微粒。",
        "11.驅蚊水並不能驅蚊，而是能干擾蚊子的感覺器官，這樣牠們就找不到人在哪裡。",
        "12.將鹽放入水中，水平面不會因加入新的物質而升高，反而會因水分子進入吸水性物質分子內填補空缺，而使得水平面下降。",
        "13.白米中的B族維生素溶於水，很可能會隨著洗米水流失。",
        "14.保齡球只需傾斜7.5度就足以倒下。",
        "15.一邊嚼口香糖一邊切洋蔥，就不會流淚。",
        "16.臭屁和臭雞蛋含有同一種發出臭味的氣體硫化氫。",
        "17.插花時，在水裡滴上一滴洗潔精，可以維持好幾天。",
        "18.吹風機對著標籤吹，等到標籤的膠熱了，就可以很容易把標籤撕下來。",
        "19.7-ELEVEn它的招牌上面的n是小寫。",
        "20.女性穿高跟鞋不僅僅是增加身高，也是改變重心，使自己顯得更加前凸後翹。",
        "21.吸管穿過拉環插進易開罐，就不會浮上來了。",
        "22.牛奶巧克力比可口可樂的咖啡因還要多。",
        "23.巧克力對於狗來說是致命的，只要幾盎司就可以使一隻小狗喪命，因為心臟神經系統受損而死亡。",
        "24.一個人一天平均走兩萬步，一年要走七百萬步。人活七十歲的話，加起來要走五億步，即三十八萬四千公里。這個數字，正好是地球到月球的距離。",
        "25.我們所穿的衣服每天把成千上萬個表皮細胞揩掉。人們其實每27天就換上一層新的表皮。",
        "26.人眼的直徑總是大致相等的——24毫米，而且幾乎不因人們的年齡而改變，所以，孩子們的眼睛看起來似乎大些。",
        "27.1977年6月25日，美國佛羅里達州的特德·聖馬田，在一次籃球表演賽中，站在罰球線上投籃，連續命中2036次。",
        "28.指甲生長速度夏天比冬天快1/5，白天比晚間快一倍，中指指甲生長速度最快。",
        "29.通過顯微鏡，我們可以看出：蚊子有22只牙齒。(真的很讓人吃驚啊，它竟然還長牙了!)",
        "30.鯨的心臟每小時只跳540次。",
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
        model="models/gemini-embedding-001"
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
    # Step 3: 測試相似度檢索功能 (Similarity Search)
    # -------------------------------------------------------------------------
    print("\n【Step 3】測試相似度檢索功能 (Similarity Search)")
    print("-" * 50)
    test_query = "蚊子有幾隻牙齒？"
    print(f"🔍 檢索測試問題：「{test_query}」")
    test_docs = vectorstore.similarity_search(test_query, k=2)
    print("檢索到的相關文檔：")
    for i, doc in enumerate(test_docs, 1):
        print(f"  [{i}] {doc.page_content}")

    # -------------------------------------------------------------------------
    # Step 4: 整合 LLM 生成答案 (自訂 Prompt + Context)
    # -------------------------------------------------------------------------
    print("\n【Step 4】整合 Google Gemini 生成答案 (Prompt + Context)")
    print("-" * 50)

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.2
    )

    def ask_rag(query: str, k: int = 3):
        """檢索相關文檔並交由 Gemini 回答"""
        docs = vectorstore.similarity_search(query, k=k)
        context = "\n\n".join(d.page_content for d in docs)
        prompt = f"""請根據以下參考資料回答問題。如果參考資料中沒有相關資訊，請誠實地說不知道。

參考資料：
{context}

問題：{query}

答案："""
        response = llm.invoke(prompt)
        return response.content, docs

    demo_query = "蚊子牙齒數跟鯨魚每分鐘心跳數哪個多?"
    print(f"❓ 示範問題：{demo_query}")
    demo_ans, demo_sources = ask_rag(demo_query)
    print(f"💡 答案：\n{demo_ans}\n")
    print("📚 參考來源：")
    for i, doc in enumerate(demo_sources, 1):
        print(f"  [{i}] {doc.page_content}")

    # -------------------------------------------------------------------------
    # 互動式問答體驗
    # -------------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("💬 進入即時問答模式（直接輸入問題按 Enter；輸入 'exit' 或 'q' 結束）：")
    print("=" * 65)

    while True:
        try:
            user_input = input("\n請輸入您的問題 > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n👋 感謝使用，再見！")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "q"):
            print("👋 感謝使用，再見！")
            break

        ans, sources = ask_rag(user_input)
        print(f"\n💡 回答：\n{ans}")
        print("\n📚 引用文檔：")
        for doc in sources:
            print(f"  - {doc.page_content}")


if __name__ == "__main__":
    main()
