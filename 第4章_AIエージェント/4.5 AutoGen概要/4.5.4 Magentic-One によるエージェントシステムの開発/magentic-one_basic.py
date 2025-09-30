# 標準ライブラリ
import os
import asyncio
# サードパーティライブラリ
from dotenv import load_dotenv
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import MagenticOneGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.agents.web_surfer import MultimodalWebSurfer
# from autogen_ext.agents.file_surfer import FileSurfer
# from autogen_ext.agents.magentic_one import MagenticOneCoderAgent
# from autogen_agentchat.agents import CodeExecutorAgent
# from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor

load_dotenv()

model_client = AzureOpenAIChatCompletionClient(
    azure_deployment=os.environ.get("AZURE_OPENAI_LLM", ""),
    model="gpt-4o",
    api_version=os.environ.get("OPENAI_API_VERSION", ""),
    azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT", ""),
    api_key=os.environ.get("AZURE_OPENAI_API_KEY", "")
)

async def main() -> None:

    surfer = MultimodalWebSurfer(
        "WebSurfer",
        model_client=model_client,
    )

    team = MagenticOneGroupChat([surfer], model_client=model_client)
    await Console(team.run_stream(task="What is the UV index in Melbourne today? "))

    #　他のエージェントもチームに加える場合はこちらを有効に
    # team = MagenticOneGroupChat([surfer, file_surfer, coder, terminal], model_client=model_client)
    # file_surfer = FileSurfer( "FileSurfer",model_client=model_client)
    # coder = MagenticOneCoderAgent("Coder",model_client=model_client)
    # terminal = CodeExecutorAgent("ComputerTerminal",code_executor=LocalCommandLineCodeExecutor())

asyncio.run(main())
