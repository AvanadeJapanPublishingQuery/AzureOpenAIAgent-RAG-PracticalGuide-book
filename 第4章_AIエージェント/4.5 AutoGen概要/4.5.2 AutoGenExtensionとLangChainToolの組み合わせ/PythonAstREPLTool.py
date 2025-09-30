# 必要なライブラリのインポート
# 標準ライブラリ
import os
import asyncio
# サードパーティライブラリ
import pandas as pd
from dotenv import load_dotenv
from langchain_experimental.tools.python.tool import PythonAstREPLTool
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from autogen_ext.tools.langchain import LangChainToolAdapter

# 環境変数の読み込み（APIキーやエンドポイント情報）
from dotenv import load_dotenv
load_dotenv()

# Azure OpenAI モデルクライアントの設定
model_client = AzureOpenAIChatCompletionClient(
    azure_deployment=os.environ.get("OPENAI_API_GPT4_OMNI_128K_20240806", ""),
    model="gpt-4o",
    api_version=os.environ.get("OPENAI_API_VERSION", ""),
    azure_endpoint=os.environ.get("OPENAI_API_BASE_URL", ""),
    api_key=os.environ.get("OPENAI_API_KEY", "")
)

# LangChain ツールの定義：Pandasデータフレーム（Titanicデータ）を扱える Python 実行環境を提供
# TODO: 実際のCSVファイルパスに置き換えてください
df = pd.read_csv("<YOUR_PATH>/titanic.csv")
tool = LangChainToolAdapter(PythonAstREPLTool(locals={"df": df}))

async def main() -> None:
    # エージェントの定義：データ分析タスクに対応する専門家エージェント
    agent = AssistantAgent(
        name="agent",
        model_client=model_client,
        description="あなたはツールを使用してユーザーを支援するエージェント。df 変数を使用してデータセットにアクセスする",
        system_message="""
            ツールを使ってタスクを解決することに長けており、データセットにアクセスする際は df 変数を使用します。
            タスクが完了したら、「[stop]」と返信してください。
            あなたはデータ分析の専門家です。
            与えられたデータを用い、最初にビジネス背景と目標を整理し、
            次にデータをクリーニングし、特徴量エンジニアリング後に探索的分析を行い、
            必要に応じてモデルを構築・評価してください。
            その結果を基に洞察、結論、ビジネスインサイト、改善策を提示し、今後の方針と検証項目を提案してください。
            分析内容を要約し、ビジネスへの影響と効果を示すことで、改善と意思決定に活かしてください。
        """,
        tools=[tool],
    )

    # 終了条件の設定：「[stop]」というフレーズが出たら対話を終了する
    termination = TextMentionTermination("[stop]")

    # エージェントチームの定義：今回は1人構成、ラウンドロビン形式で会話を進行
    agent_team = RoundRobinGroupChat(participants=[agent], termination_condition=termination, max_turns=None)

    # タスクの実行：Console を使って対話の様子をリアルタイム表示
    stream = agent_team.run_stream(task="乗客の男女比はどのような分布になっていますか？さらに、生存の観点から見た男女の割合、年齢の分布などはどれくらいですか？また、なぜそのような結果になったのでしょうか。他の要因（例: 乗客のクラス、発着港など）についての影響分析を行い、生存率との相関を更に調査してください。")
    await Console(stream)

# 実行エントリーポイント
if __name__ == '__main__':
    asyncio.run(main())
