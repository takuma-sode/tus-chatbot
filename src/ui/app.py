"""
東京理科大学チャットボット - Streamlit UI
起動方法: streamlit run src/ui/app.py
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

import streamlit as st
from src.rag.naive_rag import NaiveRAG
from src.rag.hybrid_rag import HybridRAG
from src.rag.hyde_rag import HyDERAG

st.set_page_config(page_title="東京理科大学chatbot", page_icon="🎓", layout="centered")
st.title("🎓 東京理科大学チャットボット")

# サイドバー：検索手法の切り替え
with st.sidebar:
    st.header("検索設定")
    mode = st.radio(
        "検索手法",
        options=[
            "ナイーブRAG（フェーズ1）",
            "ハイブリッド検索（フェーズ2）",
            "HyDE + ハイブリッド（フェーズ3）",
        ],
        index=2,
    )
    st.divider()
    if mode == "ナイーブRAG（フェーズ1）":
        st.caption("ベクトル検索のみ（上位3件）")
    elif mode == "ハイブリッド検索（フェーズ2）":
        st.caption("BM25 + ベクトル検索 → RRFで統合（各上位10件 → 上位3件）")
    else:
        st.caption("LLMで仮回答生成 → 仮回答でベクトル検索 + BM25 → RRFで統合")

st.caption(f"現在のモード: {mode}")


@st.cache_resource
def load_naive():
    return NaiveRAG()


@st.cache_resource
def load_hybrid():
    return HybridRAG()


@st.cache_resource
def load_hyde():
    return HyDERAG()


# モードに応じてRAGを選択
if mode == "ナイーブRAG（フェーズ1）":
    rag = load_naive()
elif mode == "ハイブリッド検索（フェーズ2）":
    rag = load_hybrid()
else:
    rag = load_hyde()

# モード切り替え時はチャット履歴をリセット
if "current_mode" not in st.session_state:
    st.session_state.current_mode = mode
if st.session_state.current_mode != mode:
    st.session_state.messages = []
    st.session_state.current_mode = mode

if "messages" not in st.session_state:
    st.session_state.messages = []


def render_sources(sources: list[dict], hypothesis: str | None = None):
    """参照Q&Aと仮回答を表示"""
    if hypothesis:
        with st.expander("HyDE仮回答（検索に使用）"):
            st.write(hypothesis)

    with st.expander(f"参照Q&A（{len(sources)}件）"):
        for i, src in enumerate(sources, 1):
            if "bm25_score" in src:
                score_label = f"RRFスコア: {src['score']}"
            else:
                score_label = f"類似度: {src['score']}"
            st.markdown(f"**{i}. {score_label}**")
            if src.get("vector_score") is not None:
                st.caption(f"ベクトル類似度: {src['vector_score']} / BM25スコア: {src.get('bm25_score', 'N/A')}")
            st.markdown(f"Q: {src['question']}")
            st.markdown(f"A: {src['answer']}")
            if i < len(sources):
                st.divider()


# チャット履歴を表示
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg["role"] == "assistant" and "sources" in msg:
            render_sources(msg["sources"], msg.get("hypothesis"))

# 入力
if question := st.chat_input("東京理科大学について質問してください"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("検索・回答生成中..."):
            result = rag.query(question)

        st.write(result["answer"])
        render_sources(result["sources"], result.get("hypothesis"))

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"],
        "hypothesis": result.get("hypothesis"),
    })
