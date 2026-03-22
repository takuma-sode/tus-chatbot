"""
フェーズ3: HyDE + ハイブリッド検索
- HyDE: Ollamaで仮回答を生成 → 仮回答をベクトル化して検索
- BM25: 元の質問でキーワード検索（上位10件）
- ベクトル検索: 仮回答ベクトルで検索（上位10件）
- RRF（Reciprocal Rank Fusion）で統合 → 上位3件
- Ollama gemma3:4bで回答生成
"""

import chromadb
import ollama
import pandas as pd
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

CSV_PATH = "data/chatbot_data_IT_service_desk.csv"
CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "tus_qa"
EMBED_MODEL = "intfloat/multilingual-e5-large"
LLM_MODEL = "gemma3:4b"
TOP_K = 3
BM25_CANDIDATES = 10
VECTOR_CANDIDATES = 10
RRF_K = 60

HYDE_PROMPT = """\
あなたは東京理科大学の情報に詳しいアシスタントです。
以下の質問に対して、大学の公式情報として想定される簡潔な回答を1〜2文で生成してください。
実際のデータベース検索に使用するため、キーワードを含む自然な文章にしてください。
質問：{question}
"""

ANSWER_PROMPT = """\
あなたは東京理科大学の情報案内アシスタントです。
以下の参考情報をもとに、質問に対して正確・簡潔に日本語で回答してください。
参考情報にない内容は「情報が見つかりませんでした」と答えてください。

参考情報：
{context}

質問：{question}
"""


def tokenize(text: str) -> list[str]:
    """文字2-gramでトークン化（日本語対応、追加ライブラリ不要）"""
    text = text.lower()
    return [text[i:i+2] for i in range(len(text) - 1)] or list(text)


class HyDERAG:
    def __init__(self):
        print("埋め込みモデルを読み込み中...")
        self.embed_model = SentenceTransformer(EMBED_MODEL)

        print("ChromaDBを接続中...")
        self.client = chromadb.PersistentClient(path=CHROMA_PATH)
        self.collection = self.client.get_collection(COLLECTION_NAME)

        print("BM25インデックスを構築中...")
        df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
        self.qa_data = [
            {"question": row["Question"], "answer": row["Answer"]}
            for _, row in df.iterrows()
        ]
        corpus = [tokenize(item["question"]) for item in self.qa_data]
        self.bm25 = BM25Okapi(corpus)
        print("初期化完了")

    def _generate_hypothesis(self, question: str) -> str:
        """HyDE: 質問に対する仮回答をLLMで生成"""
        prompt = HYDE_PROMPT.format(question=question)
        response = ollama.chat(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        return response["message"]["content"]

    def _vector_search(self, text: str) -> list[dict]:
        """ベクトル検索: 仮回答テキストで上位VECTOR_CANDIDATES件を返す"""
        query_vec = self.embed_model.encode(
            ["query: " + text], normalize_embeddings=True
        ).tolist()
        results = self.collection.query(
            query_embeddings=query_vec,
            n_results=VECTOR_CANDIDATES,
        )
        hits = []
        for rank in range(len(results["ids"][0])):
            meta = results["metadatas"][0][rank]
            score = 1 - results["distances"][0][rank]
            hits.append({
                "question": meta["question"],
                "answer": meta["answer"],
                "vector_rank": rank,
                "vector_score": round(score, 4),
            })
        return hits

    def _bm25_search(self, question: str) -> list[dict]:
        """BM25検索: 元の質問で上位BM25_CANDIDATES件を返す"""
        tokens = tokenize(question)
        scores = self.bm25.get_scores(tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:BM25_CANDIDATES]
        hits = []
        for rank, idx in enumerate(top_indices):
            hits.append({
                "question": self.qa_data[idx]["question"],
                "answer": self.qa_data[idx]["answer"],
                "bm25_rank": rank,
                "bm25_score": round(float(scores[idx]), 4),
            })
        return hits

    def _rrf_merge(self, vector_hits: list[dict], bm25_hits: list[dict]) -> list[dict]:
        """RRFで2つのランキングを統合"""
        rrf_scores: dict[str, dict] = {}

        for hit in vector_hits:
            key = hit["question"]
            if key not in rrf_scores:
                rrf_scores[key] = {"data": hit, "rrf": 0.0}
            rrf_scores[key]["rrf"] += 1.0 / (RRF_K + hit["vector_rank"] + 1)

        for hit in bm25_hits:
            key = hit["question"]
            if key not in rrf_scores:
                rrf_scores[key] = {"data": hit, "rrf": 0.0}
            rrf_scores[key]["rrf"] += 1.0 / (RRF_K + hit["bm25_rank"] + 1)

        sorted_items = sorted(rrf_scores.values(), key=lambda x: x["rrf"], reverse=True)

        results = []
        for item in sorted_items[:TOP_K]:
            d = item["data"]
            results.append({
                "question": d["question"],
                "answer": d["answer"],
                "score": round(item["rrf"], 6),
                "vector_score": d.get("vector_score"),
                "bm25_score": d.get("bm25_score"),
            })
        return results

    def search(self, question: str) -> tuple[list[dict], str]:
        """HyDE + ハイブリッド検索。検索結果と仮回答を返す"""
        hypothesis = self._generate_hypothesis(question)
        vector_hits = self._vector_search(hypothesis)
        bm25_hits = self._bm25_search(question)
        hits = self._rrf_merge(vector_hits, bm25_hits)
        return hits, hypothesis

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
        hits, hypothesis = self.search(question)
        answer = self.generate(question, hits)
        return {"answer": answer, "sources": hits, "hypothesis": hypothesis}
