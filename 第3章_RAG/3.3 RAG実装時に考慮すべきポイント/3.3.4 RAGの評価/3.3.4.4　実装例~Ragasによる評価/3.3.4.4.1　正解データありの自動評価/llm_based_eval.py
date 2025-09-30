import os 
from dotenv import load_dotenv 
from ragas import SingleTurnSample 
from ragas.metrics import LLMContextPrecisionWithReference 

load_dotenv() 

evaluator_llm = LangchainLLMWrapper(AzureChatOpenAI( 
    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"], # Azure OpenAI API バージョン 
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"], # Azure OpenAI エンドポイント 
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"], # Azure OpenAI デプロイメント名 
    model=os.environ["AZURE_OPENAI_MODEL_NAME"], # Azure OpenAI モデル名 
)) 

context_precision = LLMContextPrecisionWithReference(llm=evaluator_llm) 
sample = SingleTurnSample( 
    user_input="東京タワーの高さを教えてください。", 
    reference="東京タワーの高さは333メートルです。", 
    retrieved_contexts=[ 
        "高さは333mである東京の有名なランドマークといえば、東京タワーです。" 
    ]) 
await context_precision.single_turn_ascore(sample) 