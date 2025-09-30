import os
import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from autogen_agentchat.ui import Console
from dotenv import load_dotenv

load_dotenv()

# OpenAI モデルクライアントの初期化
model_client = AzureOpenAIChatCompletionClient(
    azure_deployment=os.environ.get("AZURE_OPENAI_LLM", ""),
    model="gpt-4o",
    api_version=os.environ.get("OPENAI_API_VERSION", ""),
    azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT", ""),
    api_key=os.environ.get("AZURE_OPENAI_API_KEY", "")
)

# --- 四則演算ツールの定義 ---
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

# --- メイン関数 ---
async def main() -> None:
    calc_agent = AssistantAgent(
        name="keisan_agent",
        model_client=model_client,
        description="四則演算（足し算、引き算、掛け算、割り算）を行うことができるエージェントです。",
        system_message=(
            "あなたは計算専門のアシスタントです。"
            "ユーザーが数値と演算を指定した場合、適切なツール（add, subtract, multiply, divide）を使って計算してください。"
            "タスクが完了したら [Stop] と返答してください。"
        ),
        tools=[add, subtract, multiply, divide],
    )

    termination = TextMentionTermination("Stop")

    agent_team = RoundRobinGroupChat(
        participants=[calc_agent],
        termination_condition=termination,
        max_turns=None
    )

    stream = agent_team.run_stream(task="25 x4 + 10 ÷ 2 - 5 = ?")
    await Console(stream)

if __name__ == '__main__':
    asyncio.run(main())
