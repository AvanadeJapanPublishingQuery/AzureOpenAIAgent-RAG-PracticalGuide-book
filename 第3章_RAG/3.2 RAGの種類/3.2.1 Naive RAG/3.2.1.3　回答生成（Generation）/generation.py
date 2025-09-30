# 回答生成（Generation）

# 1. プロンプト作成
system_prompt = ChatCompletionSystemMessageParam(
    content="あなたは、優秀なAIアシスタントです。",
    role="system"
)

# 2. クエリの結果をプロンプトに追加

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

api_key = os.environ["AZURE_OPENAI_API_KEY"]
api_version = os.environ["AZURE_OPENAI_API_VERSION"]
azure_endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
model = os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"]



azure_openai_client = openai.AzureOpenAI(
    api_key=api_key,
    api_version=api_version,
    azure_endpoint=azure_endpoint
)

# 3. LLMへのリクエスト
response = azure_openai_client.chat.completions.create(
    model=model,
    messages=[system_prompt, user_prompt],
    temperature=0.0
)

# 4. 回答
response_text = response.choices[0].message.content
print(response_text)