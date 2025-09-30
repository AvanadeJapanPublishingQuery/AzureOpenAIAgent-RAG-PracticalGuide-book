import json 
import os 
import pandas as pd 
from sklearn.metrics.pairwise import cosine_similarity 
import openai 

# Azure OpenAI の API 設定 

api_key = os.environ["AZURE_OPENAI_API_KEY"]# APIキー 
api_version = os.environ["AZURE_OPENAI_API_VERSION"]# APIバージョン 
azure_endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]# エンドポイントURL 
embedding_model = os.environ["AZURE_OPENAI_EMBEDDING_MODEL_DEPLOYMENT_NAME"]# ベクトル化モデルのデプロイ名 
chat_model = os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"]# チャットモデルのデプロイ名 