# 東京理科大学チャットボット

東京理科大学のQ&Aデータを使ったRAGチャットボット。
完全ローカルで動作し、外部APIへのデータ送信は一切行いません。

## 技術スタック

| 役割 | 技術 |
|------|------|
| UI | Streamlit |
| 埋め込みモデル | intfloat/multilingual-e5-large（ローカル） |
| ベクトルDB | ChromaDB（ローカル） |
| キーワード検索 | BM25（rank_bm25） |
| LLM | Ollama + gemma3:4b（ローカル） |

## 検索手法

3つの手法をUIから切り替えて使えます。

- **フェーズ1 - ナイーブRAG**: ベクトル検索のみで上位3件を取得
- **フェーズ2 - ハイブリッド検索**: BM25 + ベクトル検索をRRFで統合
- **フェーズ3 - HyDE + ハイブリッド**: LLMで仮回答を生成し、仮回答でベクトル検索 + BM25をRRFで統合

## 動作環境

- Python 3.10以上
- [Ollama](https://ollama.com/) インストール済み
- gemma3:4bモデル取得済み（`ollama pull gemma3:4b`）

## セットアップ

### 1. リポジトリのクローン

```bash
git clone https://github.com/takuma-sode/tus-chatbot.git
cd tus-chatbot
```

### 2. 依存ライブラリのインストール

```bash
pip install -r requirements.txt
```

### 3. データの配置

`data/`フォルダを作成し、Q&AデータのCSVファイルを配置してください。

```
data/
└── chatbot_data_IT_service_desk.csv
```

CSVのカラム構成: `Question, Answer`

> **注意**: データは非公開のため、このリポジトリには含まれていません。

### 4. ChromaDBへのデータ投入

```bash
python -m src.ingest
```

## 起動方法

```bash
streamlit run src/ui/app.py
```

ブラウザで http://localhost:8501 が開きます。

## ファイル構成

```
tus-chatbot/
├── README.md
├── requirements.txt
├── data/                   # Q&Aデータ（非公開・要別途用意）
├── chroma_db/              # ベクトルDB（自動生成）
└── src/
    ├── ingest.py           # ChromaDBへのデータ投入スクリプト
    ├── rag/
    │   ├── naive_rag.py    # フェーズ1: ナイーブRAG
    │   ├── hybrid_rag.py   # フェーズ2: ハイブリッド検索
    │   └── hyde_rag.py     # フェーズ3: HyDE + ハイブリッド
    └── ui/
        └── app.py          # Streamlit UI
```
