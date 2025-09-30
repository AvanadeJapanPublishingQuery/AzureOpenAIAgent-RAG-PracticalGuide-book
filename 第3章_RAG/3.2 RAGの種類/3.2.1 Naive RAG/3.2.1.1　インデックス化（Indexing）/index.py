import os
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes.models import (
    SearchIndex, SimpleField, SearchFieldDataType, SearchableField, SearchField, VectorSearch, HnswAlgorithmConfiguration, VectorSearchProfile
)
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from dotenv import load_dotenv
import openai
from openai.types.chat.chat_completion_system_message_param import ChatCompletionSystemMessageParam
from openai.types.chat.chat_completion_user_message_param import ChatCompletionUserMessageParam

load_dotenv()

# インデックス化（Indexing）

# 1.データのロード
with open("../data/data_shishin.pdf", "rb") as file:
    document_bytes = file.read()


# 2.Document Intelligence クライアントの作成
client = DocumentAnalysisClient(
    endpoint=os.environ["AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT"],
    credential=AzureKeyCredential(os.environ["AZURE_DOCUMENT_INTELLIGENCE_KEY"])
)

# 3.データからテキスト抽出
result = client.begin_analyze_document(model_id=os.environ["AZURE_DOCUMENT_INTELLIGENCE_MODEL"], document=document_bytes).result()

# 4.テキストのチャンク化
text_chunks = []
for page in result.pages:
    page_text = "\n".join([line.content for line in page.lines])
    # 適当な長さで分割（ここでは300文字ごと）
    chunk_size = 300
    chunks = [page_text[i:i+chunk_size] for i in range(0, len(page_text), chunk_size)]
    text_chunks.extend(chunks)


# 5.チャンクのベクトル化
api_key = os.environ["AZURE_OPENAI_EMBEDDING_API_KEY"]
api_version = os.environ["AZURE_OPENAI_EMBEDDING_API_VERSION"]
azure_endpoint = os.environ["AZURE_OPENAI_EMBEDDING_ENDPOINT"]
model = os.environ["AZURE_OPENAI_EMBEDDING_MODEL_DEPLOYMENT_NAME"]

azure_openai_client = openai.AzureOpenAI(
    api_key=api_key,
    api_version=api_version,
    azure_endpoint=azure_endpoint
)

embedding_results = []

for text in text_chunks:
    response = azure_openai_client.embeddings.create(
        model=model,
        input=[text]  # 1つずつ渡す
    )
    embedding_results.append(response.data[0].embedding)


# 6.Azure AI Search Index クライアントの作成
service_endpoint = os.environ["AZURE_SEARCH_SERVICE_ENDPOINT"]
index_name = os.environ["AZURE_SEARCH_INDEX_NAME"]
key = os.environ["AZURE_SEARCH_API_KEY"]
search_index_client = SearchIndexClient(
    endpoint=service_endpoint,
    index_name=index_name,
    credential=AzureKeyCredential(key)
)

# 7.インデックスの作成

fields = [
    SimpleField(name="id", type=SearchFieldDataType.String, key=True),
    SearchableField(name="content", type=SearchFieldDataType.String),
    SearchField(
    name="embedding",
    type=SearchFieldDataType.Collection(SearchFieldDataType.Single), 
    searchable=True,
    vector_search_dimensions=1536,
    vector_search_profile_name="hnsw_profile"
)
]

index = SearchIndex(
    name=index_name,
    fields=fields,
    vector_search=VectorSearch(
        algorithms=[
            HnswAlgorithmConfiguration(name="hnsw_algorithm")
        ],
        profiles=[
            VectorSearchProfile(name="hnsw_profile", algorithm_configuration_name="hnsw_algorithm")
        ]
    )
)

search_index_client.create_or_update_index(index)

# 8.データのインデックス化

search_client = SearchClient(
    endpoint=service_endpoint,
    index_name=index_name,
    credential=AzureKeyCredential(key)
)

documents = [
    {"id": str(i), "content": text_chunks[i], "embedding": list(embedding_results[i])}
    for i in range(len(text_chunks))
]

search_client.upload_documents(documents)