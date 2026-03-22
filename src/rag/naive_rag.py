"""
フェーズ1: ナイーブRAG
- questionをベクトル検索（ChromaDB）で上位3件取得
- Ollama gemma3:4bで回答生成
"""

import chromadb
import ollama
from sentence_transformers import SentenceTransformer

CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "tus_qa"
EMBED_MODEL = "intfloat/multilingual-e5-large"
LLM_MODEL = "gemma3:4b"
TOP_K = 3

ANSWER_PROMPT = """\
あなたは東京理科大学の情報案内アシスタントです。
以下の参考情報をもとに、質問に対して正確・簡潔に日本語で回答してください。
参考情報にない内容は「情報が見つかりませんでした」と答えてください。

参考情報：
{context}

質問：{question}
"""


class NaiveRAG:
    def __init__(self):
        print("埋め込みモデルを読み込み中...")
        self.model = SentenceTransformer(EMBED_MODEL)
        self.client = chromadb.PersistentClient(path=CHROMA_PATH)
        self.collection = self.client.get_collection(COLLECTION_NAME)

    def search(self, question: str) -> list[dict]:
        query_vec = self.model.encode(
            ["query: " + question], normalize_embeddings=True
        ).tolist()

        results = self.collection.query(
            query_embeddings=query_vec,
            n_results=TOP_K,
        )

        hits = []
        for i in range(len(results["ids"][0])):
            meta = results["metadatas"][0][i]
            score = 1 - results["distances"][0][i]  # cosine距離 → 類似度
            hits.append({
                "question": meta["question"],
                "answer": meta["answer"],
                "score": round(score, 4),
            })
        return hits

    def generate(self, question: str, hits: list[dict]) -> str:
        context = "\n\n".join(
            f"Q: {h['question']}\nA: {h['answer']}" for h in hits
        )
        prompt = ANSWER_PROMPT.format(context=context, question=question)
        response = ollama.chat(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        return response["message"]["content"]

    def query(self, question: str) -> dict:
        hits = self.search(question)
        answer = self.generate(question, hits)
        return {"answer": answer, "sources": hits}
