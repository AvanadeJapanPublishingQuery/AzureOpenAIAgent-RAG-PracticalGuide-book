# --------------------------
# 1. 必要なライブラリのインポート
# --------------------------
import os
from pathlib import Path
import requests
import pykakasi

from llama_index.core import (
    SimpleDirectoryReader, 
    VectorStoreIndex, 
    SummaryIndex, 
    load_index_from_storage, 
    StorageContext
)
from llama_index.core.objects import ObjectIndex
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.agent.openai import OpenAIAgent
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.core import Settings

# --------------------------
# 2. ウィキペディア記事のタイトルを設定
# --------------------------
wiki_titles = ["清水寺", "東大寺", "法隆寺", "中尊寺"]

# --------------------------
# 3. ウィキペディアAPIから記事を取得し、ローカルに保存
# --------------------------
data_path = Path("data")
data_path.mkdir(exist_ok=True)  # フォルダがない場合は作成

for title in wiki_titles:
    response = requests.get(
        "https://ja.wikipedia.org/w/api.php",
        params={
            "action": "query",
            "format": "json",
            "titles": title,
            "prop": "extracts",
            "explaintext": True,
        },
    ).json()
    page = next(iter(response["query"]["pages"].values()))
    wiki_text = page.get("extract", "")

    with open(data_path / f"{title}.txt", "w", encoding="utf-8") as fp:
        fp.write(wiki_text)

# --------------------------
# 4. テキストファイルを読み込み
# --------------------------
temple_docs = {
    wiki_title: SimpleDirectoryReader(input_files=[f"data/{wiki_title}.txt"]).load_data()
    for wiki_title in wiki_titles
}

# --------------------------
# 5. Azure OpenAIの設定（APIキーは環境変数から取得）
# --------------------------
openai_chat_model = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("OPENAI_API_VERSION"),
    deployment_name=os.getenv("AZURE_OPENAI_LLM "),
    model="gpt-4o",
)
Settings.llm = openai_chat_model

openai_embedding_model = AzureOpenAIEmbedding(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("OPENAI_API_VERSION"),
    deployment_name=os.getenv("AZURE_OPENAI_EMBEDDING "),
    model="text-embedding-ada-002"
)
Settings.embed_model = openai_embedding_model
# --------------------------
# 6. 各寺院用エージェントを作成
# --------------------------
agents = {}
query_engines = {}

for wiki_title in wiki_titles:
    # テキストをノード（小さいチャンク）に分割
    nodes = temple_docs[wiki_title]
    
    # ベクターインデックスの作成
    vector_index = VectorStoreIndex(nodes)
    vector_query_engine = vector_index.as_query_engine(llm=Settings.llm)

    # サマリーインデックスの作成
    summary_index = SummaryIndex(nodes)
    summary_query_engine = summary_index.as_query_engine(llm=Settings.llm)

    # クエリエンジンの設定
    query_engine_tools = [
QueryEngineTool(
query_engine=vector_query_engine,
metadata=ToolMetadata(
name="vector_tool",
description=f"{wiki_title}の詳細情報検索ツール"
)
),
QueryEngineTool(
query_engine=summary_query_engine,
metadata=ToolMetadata(
name="summary_tool",
description=f"{wiki_title}の要約ツール"
)
)
]


    # エージェントの作成
    agent = OpenAIAgent.from_tools(query_engine_tools, llm=Settings.llm, verbose=True)
    agents[wiki_title] = agent

# --------------------------
# 7. 上位エージェントを作成（複数の寺院を統括）
# --------------------------
all_tools = [
    QueryEngineTool(query_engine=agents[wiki_title], metadata=ToolMetadata(name=f"tool_{wiki_title}", description=f"{wiki_title}に関する質問に対応するツール"))
    for wiki_title in wiki_titles
]

obj_index = ObjectIndex.from_objects(all_tools, index_cls=VectorStoreIndex)

top_agent = ReActAgent.from_tools(
    llm=Settings.llm,
    max_iterations=50,
    tool_retriever=obj_index.as_retriever(similarity_top_k=5),
    system_prompt="あなたは日本の寺院に関する質問に対応するAIエージェントです。",
    verbose=True,
)

# --------------------------
# 8. 質問を試してみる
# --------------------------
response = top_agent.query("以下の寺院を創建年の古い順に正しく並べ替えよ。清水寺、東大寺、法隆寺、中尊寺、薬師寺")
print(response)
