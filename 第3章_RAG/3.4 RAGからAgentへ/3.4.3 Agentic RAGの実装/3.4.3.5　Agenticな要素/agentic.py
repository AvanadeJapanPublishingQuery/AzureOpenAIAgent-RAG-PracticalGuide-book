functions_map = {"retrieve_documents": retrieve_documents}
tools = [
    {"type": "function",
        "function": {
            "name": "retrieve_documents",
            "description": "ユーザーの質問に基づいて関連する文書を検索する",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query",
                    }
                },
                "required": ["query"]
        },
        }
    }
]
# エージェントのように振る舞う関数
def agentic_search(message: str, preamble: str, verbose: bool = False) -> str:
    chat_history = [{"role": "system", "content": preamble}, {"role": "user", "content": message}]
    while True:
        response = azure_openai_client.chat.completions.create(model=chat_model, messages=chat_history, tools=tools)
        tool_calls = response.choices[0].message.tool_calls
        if not tool_calls:
            return response.choices[0].message.content  # ツール呼び出しなしなら終了
        tool_results = [{"call": tool_call.function.name, "output": functions_map[tool_call.function.name](**json.loads(tool_call.function.arguments))}
                        for tool_call in tool_calls]
        chat_history.extend([
            {"role": "assistant", "content": None, "tool_calls": tool_calls},
            *[{"role": "tool", "tool_call_id": tool_call.id, "content": json.dumps(result["output"])} for tool_call, result in zip(tool_calls, tool_results)]
        ])
        if verbose:
            print(f"Tool Results: {tool_results}")
        # ループを抜けるための `break`
        if not response.choices[0].message.tool_calls:
            break  # もうツールの呼び出しがなければ終了
        system_prompt = """
        あなたは社内ポリシーに関する質問に答える専門アシスタントです。
        以下のルールを厳密に守ってください：
        1. 質問に答えるには、必ず `retrieve_documents` ツールを使用して文書を取得してください。
        2. 取得した文書の中に「◯◯を参照してください」「詳細は△△に記載」など、他の文書への言及が含まれている場合は、その文書名（またはセクション名）を使って **もう一度 `retrieve_documents` ツールを呼び出してください**。
        3. 情報が十分に揃っていると判断できるまで、この再検索を繰り返してください。
        4. 十分な情報が揃った段階で、正確かつ簡潔に質問に回答してください。
        5. 回答には、検索結果に含まれる情報 **のみ** を使用してください。推測や外部知識を加えないでください。
        """
