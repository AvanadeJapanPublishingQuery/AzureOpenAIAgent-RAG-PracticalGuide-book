import asyncio
import os
from dotenv import load_dotenv

# UIおよびチーム構成に必要なモジュール
from autogen_agentchat.ui import Console
from autogen_agentchat.teams import MagenticOneGroupChat
from autogen_agentchat.agents import AssistantAgent

# OpenAIクライアント（Azure経由）
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient

# Pythonコードの実行ツール
from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor
from autogen_ext.tools.code_execution import PythonCodeExecutionTool

# Webサーフィング（情報収集）用のエージェント
from autogen_ext.agents.web_surfer import MultimodalWebSurfer

# 環境変数を読み込む
load_dotenv()

# Azure OpenAI モデルクライアントの初期化
model_client = AzureOpenAIChatCompletionClient(
    azure_deployment=os.environ.get("AZURE_OPENAI_LLM", ""),
    model="gpt-4o",
    api_version=os.environ.get("OPENAI_API_VERSION", ""),
    azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT", ""),
    api_key=os.environ.get("AZURE_OPENAI_API_KEY", "")
)

async def main() -> None:
    # Webから情報を収集するエージェント
    web_surfer = MultimodalWebSurfer(
        "WebSurfer",
        model_client=model_client,
    )

    # Pythonコードの実行環境を設定
    tool = PythonCodeExecutionTool(LocalCommandLineCodeExecutor(work_dir="coding"))

    # コーディングとレポート生成を担当するアシスタントエージェント
    code_agent = AssistantAgent(
        "assistant",
        model_client,
        tools=[tool],
        reflect_on_tool_use=True,
        system_message=(
            "You are a coding assistant. Your task is to help the user write code and generate reports. "
            "You can use Python libraries such as yfinance and matplotlib to create plots and reports. "
            "you must run the code with tool rather than just write the code." 
        ),
    )

    # チームを構成し、会話形式でタスクを実行
    team = MagenticOneGroupChat([web_surfer, code_agent], model_client=model_client)

    # タスク：Microsoft社の2024年レポートを作成する
    await Console(team.run_stream(
        task="Microsoft社の2024年の株価および主な出来事をまとめたHTMLレポートを作成してください。"
             "また、関連するグラフとデータ分析を含めてください。"
             "Pythonプログラミング言語を使用し、yfinance と matplotlib ライブラリを利用してください。"
             "グラフと文章が一体となったHTMLレポートを生成してください。"
    ))

# 非同期タスクの実行
asyncio.run(main()) 
