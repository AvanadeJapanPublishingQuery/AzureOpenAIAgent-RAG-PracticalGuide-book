embedding_results = [] 
for text in db["combined"].tolist(): 
    response = azure_openai_client.embeddings.create( 
        model=embedding_model, 
        input=[text] 
    ) 
    embedding_results.append(response.data[0].embedding) 

db['embeddings'] = embedding_results 
db.head() 

def retrieve_documents(query: str, n=1) -> dict: 
    #クエリをベクトル化 
    query_emb = azure_openai_client.embeddings.create( 
        model=embedding_model, 
        input=[query]
    )
    #コサイン類似度の計算 
    similarity_scores = cosine_similarity( 
        [query_emb.data[0].embedding], db['embeddings'].tolist() 
    )[0] 
    # 上位n件を抽出 
    top_indices = similarity_scores.argsort()[::-1][:n] 
    top_matches = db.iloc[top_indices] 
    return {"top_matched_document": top_matches.combined.tolist()}