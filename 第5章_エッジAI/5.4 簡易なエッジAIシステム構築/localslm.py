import openai
from foundry_local import FoundryLocalManager


def main():
    #　ダウンロードするモデルを変更する場合は、aliasのモデル名を変更します
    alias = "phi-4-mini"

    #　Foundry Local Serviceの起動／ロードを行う
    manager = FoundryLocalManager(alias)

    #　OpenAI Python SDKを用いて生成AIモデルと対話する
    #　base_urlにmanager.endpointと設定することでFoundry Local Serviceを使用
    #　ローカル使用においては、APIキーは必要ありません
    client = openai.OpenAI(
        base_url=manager.endpoint,
        api_key=manager.api_key, 
    )

    #　ユーザーの入力
    print("質問を入力してください: ")
    user_input = input()

    stream = client.chat.completions.create(
        model=manager.get_model_info(alias).id,
        messages=[{"role": "user", "content": user_input}],
        stream=True,
    )

    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            print(chunk.choices[0].delta.content, end="", flush=True)

if __name__ == "__main__":
    main()

