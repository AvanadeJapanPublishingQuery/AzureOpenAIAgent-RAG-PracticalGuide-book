import os
import uuid
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    VectorSearch,
    SimpleField,
    SearchFieldDataType,
    SearchField,
    SearchIndex,
    HnswAlgorithmConfiguration,
    HnswParameters,
    VectorSearchAlgorithmMetric,
    ScalarQuantizationCompression,
    ScalarQuantizationParameters,
    VectorSearchProfile,
)
import openai
from openai.types.chat.chat_completion_system_message_param import ChatCompletionSystemMessageParam
from openai.types.chat.chat_completion_user_message_param import ChatCompletionUserMessageParam


try:
    endpoint = os.environ["AZURE_AI_VISION_ENDPOINT"]
    key = os.environ["AZURE_AI_VISION_API_KEY"]
except KeyError:
    print("Please set the environment variables AZURE_AI_VISION_ENDPOINT and AZURE_AI_VISION_API_KEY")

# 画像データの前処理

# 1. イメージデータの読み込み
# 画像ファイルのリスト
image_files = ["../images/sample1.jpg", "../images/sample2.jpg"]

# 各画像を読み込んで処理する
image_data_list = []
for image_file in image_files:
    try:
        with open(image_file, "rb") as image:
            image_data = image.read()
            image_data_list.append({"filename": image_file, "data": image_data})
    except FileNotFoundError:
        print(f"File {image_file} not found. Skipping.")

# 2. イメージ解析クライアントの作成
client = ImageAnalysisClient(endpoint=endpoint, credential=AzureKeyCredential(key))

# 3. 画像のキャプションの取得
# Captionのサポートは、リージョンによって異なる。日本東部リージョンはサポートしていない。
results = []
for image in image_data_list:
    try:
        # キャプション生成は、英語のみの対応状況 https://learn.microsoft.com/en-us/azure/ai-services/computer-vision/language-support#image-analysis
        result = client.analyze(image_data=image["data"], visual_features=[VisualFeatures.CAPTION, VisualFeatures.READ])
        results.append({"filename": image["filename"], "caption": result.caption.text})
    except Exception as e:
        print(f"An error occurred while processing {image['filename']}: {e}")

# 4.キャプションのベクトル化
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

for result in results:
    try:
        response = azure_openai_client.embeddings.create(
            model=model,
            input=[result["caption"]]
        )
        embedding_results.append({
            "id": str(uuid.uuid4()),
            "filename": result["filename"],
            "caption": result["caption"],
            "captionVector": response.data[0].embedding
        })
    except Exception as e:
        print(f"An error occurred while embedding {result['filename']}: {e}")

# インデックス化（Indexing）

# 6.Azure AI Search Index クライアントの作成
service_endpoint = os.environ["AZURE_SEARCH_SERVICE_ENDPOINT"]
index_name = os.environ["AZURE_SEARCH_INDEX_NAME"]
key = os.environ["AZURE_SEARCH_API_KEY"]
search_index_client = SearchIndexClient(
    endpoint=service_endpoint,
    index_name=index_name,
    credential=AzureKeyCredential(key)
)

fields = [
    SimpleField(name="id", type=SearchFieldDataType.String, key=True),
    SimpleField(name="filename", type=SearchFieldDataType.String, searchable=True),
    SearchField(name="caption", type=SearchFieldDataType.String, searchable=True),
    SearchField(
            name="captionVector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            vector_search_dimensions=1536,
            vector_search_profile_name="hnsw_profile",
        )
]

index = SearchIndex(
    name=index_name,
    fields=fields,
    vector_search=VectorSearch(
        algorithms=[
            HnswAlgorithmConfiguration(
                name="hnsw_algorithm",
                parameters=HnswParameters(
                    m=4,
                    ef_construction=400,
                    ef_search=500,
                    metric=VectorSearchAlgorithmMetric.COSINE,
                ))
        ],
        compressions=[
            ScalarQuantizationCompression(
                compression_name="myScalarQuantization",
                rerank_with_original_vectors=True,
                default_oversampling=10,
                parameters=ScalarQuantizationParameters(quantized_data_type="int8")
            )
        ],
        profiles=[
            VectorSearchProfile(name="hnsw_profile", algorithm_configuration_name="hnsw_algorithm")
        ]
    )
)

search_index_client.create_or_update_index(index)

# 7. データのインデックス化

search_client = SearchClient(
    endpoint=service_endpoint,
    index_name=index_name,
    credential=AzureKeyCredential(key)
)

search_client.upload_documents(documents=embedding_results)

# 情報検索（Retrieval）

# 8. クエリのベクトル化
query_text = "young engineer working new school project robotics"
query_embedding = azure_openai_client.embeddings.create(
    model=model,
    input=[query_text]
).data[0].embedding

# 9. ベクトル検索

vector_query = {
    "vector": query_embedding,
    "k": 5,
    "fields": "captionVector",
    "kind": "vector",
    "profile": "hnsw_profile"
}

search_results = search_client.search(
    search_text="",
    vector_queries=[vector_query],
    select="filename,caption",
    top=1
)

if not search_results:
    print("No results found.")

for result in search_results:
    print(f"ID: {result}")

retrieved_texts = "\n\n".join(result["content"] for result in search_results)


# 回答生成（Generation）

# 10. プロンプト生成
system_prompt = ChatCompletionSystemMessageParam(
    content="あなたは、優秀なAIアシスタントです。",
    role="system"
)

# 11. クエリの結果をプロンプトに追加

user_prompt = ChatCompletionUserMessageParam(
    content=f"""
あなたは正確な情報を提供するAIアシスタントです。以下の検索結果に記載されている情報 **のみ** を使用して、「{query_text}」について説明してください。

**ルール**
1. **検索結果に記載されていない内容は、絶対に回答に含めないでください。**
2. **一般的な推測、背景知識、個人的な見解、類推による補足説明は一切行わないでください。**
3. **検索結果内に答えがない場合は、「提供された情報では十分な回答ができません」とのみ述べてください。**
4. **曖昧な表現を避け、事実ベースで簡潔に回答してください。**
5. **言い換えや解釈を行わず、検索結果の表現をできるだけ忠実に使用してください。**

**検索結果**
{retrieved_texts}

---
上記の情報をもとに、「{query_text}」について説明してください。
""",
    role="user"
)