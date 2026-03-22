"""
CSVデータをChromaDBに投入するスクリプト
Usage: python -m src.ingest
"""

import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer

CSV_PATH = "data/chatbot_data_IT_service_desk.csv"
CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "tus_qa"
EMBED_MODEL = "intfloat/multilingual-e5-large"


def main():
    print("CSVを読み込み中...")
    df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
    print(f"  {len(df)}件読み込み完了")

    print("埋め込みモデルを読み込み中...")
    model = SentenceTransformer(EMBED_MODEL)

    print("埋め込みベクトルを生成中...")
    texts = ["passage: " + q for q in df["Question"].tolist()]
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True).tolist()

    print("ChromaDBに保存中...")
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # 既存コレクションがあれば削除して再作成
    try:
        client.delete_collection(COLLECTION_NAME)
        print("  既存コレクションを削除しました")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    collection.add(
        ids=[str(i) for i in range(len(df))],
        embeddings=embeddings,
        metadatas=[
            {"question": row["Question"], "answer": row["Answer"]}
            for _, row in df.iterrows()
        ],
    )

    print(f"完了: {len(df)}件をChromaDBに保存しました")


if __name__ == "__main__":
    main()
