# 情報検索（Retrieval）

# 1. クエリのベクトル化

query_text = "オープンデータとは"
query_embedding = azure_openai_client.embeddings.create(
    model=model,
    input=[query_text]
).data[0].embedding

# 2. クエリの実行

vector_query = {
    "vector": query_embedding,
    "k": 5,
    "fields": "embedding",
    "kind": "vector",
    "profile": "hnsw_profile"
}

search_results = search_client.search(
    search_text="",
    vector_queries=[vector_query],
    select=["id", "content"],
    top=3
)

print("検索結果")
for result in search_results:
    print(f"ID: {result['id']}")
    print(f"Content: {result['content']}\n")
    print(f"Score: {result['@search.score']}\n")

retrieved_texts = "\n\n".join(result["content"] for result in search_results)
