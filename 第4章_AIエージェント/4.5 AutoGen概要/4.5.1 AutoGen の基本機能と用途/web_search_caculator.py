import os
import asyncio
import requests
from dotenv import load_dotenv

from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient

load_dotenv()

# Azure OpenAI クライアント
model_client = AzureOpenAIChatCompletionClient(
    azure_deployment=os.environ.get("AZURE_OPENAI_LLM", ""),
    model="gpt-4o",
    api_version=os.environ.get("OPENAI_API_VERSION", ""),
    azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT", ""),
    api_key=os.environ.get("AZURE_OPENAI_API_KEY", "")
)

# --- 四則演算ツール定義 ---
async def add(a: float, b: float) -> str:
    return f"{a} + {b} = {a + b}"

async def subtract(a: float, b: float) -> str:
    return f"{a} - {b} = {a - b}"

async def multiply(a: float, b: float) -> str:
    return f"{a} × {b} = {a * b}"

async def divide(a: float, b: float) -> str:
    if b == 0:
        return "エラー：ゼロ除算はできません。"
    return f"{a} ÷ {b} = {a / b}"

# --- インターネット検索ツール定義 ---
AZURE_BING_API_KEY = os.environ.get("AZURE_BING_API_KEY", "")
AZURE_BING_ENDPOINT = "https://api.bing.microsoft.com/v7.0/search"

async def web_search(query: str) -> str:
    headers = {"Ocp-Apim-Subscription-Key": AZURE_BING_API_KEY}
    params = {"q": query, "count": 20, "mkt": "ja-JP", "setLang": "JA"}
    try:
        res = requests.get(AZURE_BING_ENDPOINT, headers=headers, params=params)
        res.raise_for_status()
        data = res.json()
        items = data.get("webPages", {}).get("value", [])
        return "\n\n".join([f"{item['name']}:\n{item['snippet']}\n{item['url']}" for item in items])
    except Exception as e:
        return f"検索エラー: {e}"

# --- 各エージェント定義 ---
calculator_agent = AssistantAgent(
    name="calculator_agent",
    model_client=model_client,
    description="四則演算に特化した計算エージェントです。",
    system_message="あなたは計算専門家です。ツールを使って足し算、引き算、掛け算、割り算を正確に実行してください。完了後は [Stop] を返してください。",
    tools=[add, subtract, multiply, divide],
)

search_agent = AssistantAgent(
    name="search_agent",
    model_client=model_client,
    description="Web検索の専門家で、外部情報を収集して他のエージェントをサポートします。",
    system_message="あなたはインターネット検索専門家です。ツール 'web_search' を使って外部情報を検索し、適切に要約して提供してください。完了したら [Stop] を返してください。",
    tools=[web_search],
)


# --- チームを組んで実行 ---
async def main():
    chat = RoundRobinGroupChat(
        participants=[search_agent, calculator_agent],
        termination_condition=TextMentionTermination("Stop"),
        max_turns=8,
    )

    stream = chat.run_stream(task="三人でスカイツリー天望デッキに行きたいです。合計でいくらかかりますか？")
    await Console(stream)

if __name__ == "__main__":
    asyncio.run(main())
