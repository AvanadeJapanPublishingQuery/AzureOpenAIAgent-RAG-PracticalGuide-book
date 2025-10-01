# Azure OpenAIエージェント・RAG 構築実践ガイド ソースコードリポジトリ

本リポジトリは、書籍「Azure OpenAIエージェント・RAG 構築実践ガイド」で解説するサンプルコードおよび関連リソースを公開 / 共有するためのものです。RAG (Retrieval-Augmented Generation)、AIエージェント、エッジAI など現場実装を意識したコード例を章・節構成に対応させて配置しています。

---
## 1. リポジトリの目的
- 書籍内で参照しているコードの再現性確保
- 章 / 節ごとの差分学習・比較をしやすくする構造化
- 読者からのフィードバック (Issue / PR) を受け付け改善に反映

## 2. 書籍情報

書名: Azure OpenAIエージェント・RAG 構築実践ガイド  
Amazon: https://www.amazon.co.jp/dp/4296080466/

| 表紙 | 側面 | 裏側 |
| --- | --- | --- |
| ![表紙](./img/Azure%20OpenAIエージェント・RAG%20構築実践ガイド表紙.png) | ![側面](./img/Azure%20OpenAIエージェント・RAG%20構築実践ガイド側面.png) | ![裏側](./img/Azure%20OpenAIエージェント・RAG%20構築実践ガイド裏.png) |

| 項目 | 内容 |
| --- | --- |
| 出版社 | 日経BP |
| 出版日 | 2025/10/4 (予定) |
| 言語 | 日本語 |
| 体裁 | 単行本 |
| ページ数 | 404ページ |
| ISBN-10 | 4296080466 |
| ISBN-13 | 978-4296080465 |

## 3. 本書の主なトピック
- Agentic World と次世代 AI アプリの潮流
- Azure AI / Azure OpenAI Service 活用アーキテクチャ
- RAG 概要 / Naive・Advanced・Graph・Multi Modal など各種パターン
- 評価 (LLM / Non-LLM, Ragas 等) と品質向上手法
- Agentic RAG / マルチエージェント / MCP (Model Context Protocol)
- Azure 上でのエージェント技術 (Structured Output, Function Calling など)
- AutoGen / LangChain / LangGraph / Semantic Kernel 等との連携
- エッジAI (ローカルLLM / Microsoft Olive による最適化)
- 責任あるAI / 安全性考慮

## 4. 書籍目次
- **第1章 Agentic Worldに備えろ！**
    - 1.1 Agentic Worldへのパラダイムシフト 
    - 1.2 Agentic Worldを支える主要AI技術要素 
    - 1.3 Microsoft Azure AIサービス
- **第2章 AI Foundryおよび環境構築**
    - 2.1 Azure AI Foundryとは 
    - 2.2 豊富な機能について 
    - 2.3 環境構築
- **第3章 RAG**
    - 3.1 RAGとは 
    - 3.2 RAGの種類 
    - 3.3 RAG実装時に考慮すべきポイント 
    - 3.4 RAGからAgentへ
- **第4章 AIエージェント**
    - エージェントとは 
    - 4.1 AIエージェントの構成 
    - 4.2 AIエージェントのプロンプト構造と思考フレームワーク 
    - 4.3 Azure上でのAIエージェント技術 
    - 4.4 AutoGen概要 
    - 4.5 MCPサーバーと通信するAIエージェント
- **第5章 エッジAI**
    - 5.1 エッジAIとは 
    - 5.2 エッジAIの構成要素 
    - 5.3 エッジAIを導入する際のアーキテクチャパターン 
    - 5.4 簡易なエッジAIシステム構築 
    - 5.5 Microsoft Oliveを使ったエッジAIシステム構築 
    - 5.6 ユースケース例
- **第6章 責任あるAI**
    - 6.1 責任あるAIとは 
    - 6.2 Azure AI Content Safetyの各機能

## 5. フォルダ命名・構成方針
- 章ディレクトリ: `第n章_タイトル`
- 節ディレクトリ: 書籍内節タイトルをそのまま使用（必要に応じ半角スペース利用可）
- 各サブステップ (例: 3.2.1.1) で最小実行可能なスクリプトを配置

## 6. 実行環境の一般的前提
各コード個別 README / コメントが優先されます。共通的な前提の例:
- Python 3.10+ 推奨 (一部ライブラリ最新依存のため)
- 仮想環境利用推奨: `venv` / `uv` / `poetry`
- Azure OpenAI / Azure AI Search / Azure Storage 等のリソース権限
- 必要な API キー / エンドポイントは環境変数で注入 (例: `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_KEY`)

---
## 7. フォルダの階層
```
│  .gitignore
│  LICENSE
│  README.md
│
├─第3章_RAG
│  ├─3.2 RAGの種類
│  │  ├─3.2.1 Naive RAG
│  │  │  ├─3.2.1.1　インデックス化（Indexing）
│  │  │  │      index.py
│  │  │  │
│  │  │  ├─3.2.1.2　情報検索（Retrieval）
│  │  │  │      retrieval.py
│  │  │  │
│  │  │  └─3.2.1.3　回答生成（Generation）
│  │  │          generation.py
│  │  │
│  │  ├─3.2.2 Advanced RAG
│  │  │  └─3.2.2.2　Advanced RAGの主要コンポーネント
│  │  │          advanced_rag.py
│  │  │
│  │  ├─3.2.4 Graph RAG
│  │  │  └─3.2.4.3　実装
│  │  │          graph_rag.py
│  │  │
│  │  └─3.2.5 Multi Modal RAG
│  │      └─3.2.5.3　実装フローと各ステップの解説
│  │              multi_modal_rag.py
│  │
│  ├─3.3 RAG実装時に考慮すべきポイント
│  │  └─3.3.4 RAGの評価
│  │      └─3.3.4.4　実装例~Ragasによる評価
│  │          └─3.3.4.4.1　正解データありの自動評価
│  │                  llm_as_a_judge.py
│  │                  llm_based_eval.py
│  │                  non_llm_based_eval.py
│  │
│  └─3.4 RAGからAgentへ
│      └─3.4.3 Agentic RAGの実装
│          ├─3.4.3.2　使用ライブラリと事前準備
│          │      index.py
│          │
│          ├─3.4.3.3　データセットの準備
│          │      dataset.py
│          │
│          ├─3.4.3.4　ベクトル化と検索処理の定義
│          │      search.py
│          │
│          ├─3.4.3.5　Agenticな要素
│          │      agentic.py
│          │
│          └─3.4.3.6　実行例と出力の確認
│                  result.py
│
├─第4章_AIエージェント
│  ├─4.1 エージェントとは
│  │  └─4.1.1 Agent-Based Modeling
│  │          predator_prey.py
│  │
│  ├─4.3 エージェントのプロンプト構造と思考フレームワーク
│  │  └─4.3.2 ReActの思考フレームワーク
│  │          llama_index_react.py
│  │
│  ├─4.4 Azure上でのエージェント技術
│  │  └─4.4.1 Azure OpenAI Service の構造化出力機能
│  │          schedule_agent.py
│  │
│  └─4.5 AutoGen概要
│      ├─4.5.1 AutoGen の基本機能と用途
│      │      caculator.py
│      │      web_search_caculator.py
│      │
│      ├─4.5.2 AutoGenExtensionとLangChainToolの組み合わせ
│      │      PythonAstREPLTool.py
│      │
│      └─4.5.4 Magentic-One によるエージェントシステムの開発
│              magentic-one_advanced.py
│              magentic-one_basic.py
│
└─第5章_エッジAI
    ├─5.4 簡易なエッジAIシステム構築
    │      localslm.py
    │
    └─5.5 MICROSOFT OLIVEを使ったエッジAIシステム構築
        └─5.5.1 生成AIモデルの自動最適化
                app.py
```

## 8. 各章のソースコード
| 章-節.タイトル | リポジトリ |
| --- | --- |
| 3.2.1.1 インデックス化（Indexing） | [index.py](https://github.com/AvanadeJapanPublishingQuery/AzureOpenAIAgent-RAG-PracticalGuide-book/tree/main/第3章_RAG/3.2%20RAGの種類/3.2.1%20Naive%20RAG/3.2.1.1　インデックス化（Indexing）) |
| 3.2.1.2 情報検索（Retrieval） | [retrieval.py](https://github.com/AvanadeJapanPublishingQuery/AzureOpenAIAgent-RAG-PracticalGuide-book/tree/main/第3章_RAG/3.2%20RAGの種類/3.2.1%20Naive%20RAG/3.2.1.2　情報検索（Retrieval）) |
| 3.2.1.3 回答生成（Generation） | [generation.py](https://github.com/AvanadeJapanPublishingQuery/AzureOpenAIAgent-RAG-PracticalGuide-book/tree/main/第3章_RAG/3.2%20RAGの種類/3.2.1%20Naive%20RAG/3.2.1.3　回答生成（Generation）) |
| 3.2.2.2 Advanced RAGの主要コンポーネント | [advanced_rag.py](https://github.com/AvanadeJapanPublishingQuery/AzureOpenAIAgent-RAG-PracticalGuide-book/tree/main/第3章_RAG/3.2%20RAGの種類/3.2.2%20Advanced%20RAG/3.2.2.2　Advanced%20RAGの主要コンポーネント) |
| 3.2.4.3 Graph RAG 実装 | [graph_rag.py](https://github.com/AvanadeJapanPublishingQuery/AzureOpenAIAgent-RAG-PracticalGuide-book/tree/main/第3章_RAG/3.2%20RAGの種類/3.2.4%20Graph%20RAG/3.2.4.3　実装) |
| 3.2.5.3 Multi Modal RAG 実装フロー | [multi_modal_rag.py](https://github.com/AvanadeJapanPublishingQuery/AzureOpenAIAgent-RAG-PracticalGuide-book/tree/main/第3章_RAG/3.2%20RAGの種類/3.2.5%20Multi%20Modal%20RAG/3.2.5.3　実装フローと各ステップの解説) |
| 3.3.4.4.1 正解データありの自動評価 | [評価スクリプト群](https://github.com/AvanadeJapanPublishingQuery/AzureOpenAIAgent-RAG-PracticalGuide-book/tree/main/第3章_RAG/3.3%20RAG実装時に考慮すべきポイント/3.3.4%20RAGの評価/3.3.4.4　実装例~Ragasによる評価/3.3.4.4.1　正解データありの自動評価) |
| 3.4.3.2 使用ライブラリと事前準備 | [index.py](https://github.com/AvanadeJapanPublishingQuery/AzureOpenAIAgent-RAG-PracticalGuide-book/tree/main/第3章_RAG/3.4%20RAGからAgentへ/3.4.3%20Agentic%20RAGの実装/3.4.3.2　使用ライブラリと事前準備) |
| 3.4.3.3 データセットの準備 | [dataset.py](https://github.com/AvanadeJapanPublishingQuery/AzureOpenAIAgent-RAG-PracticalGuide-book/tree/main/第3章_RAG/3.4%20RAGからAgentへ/3.4.3.3　データセットの準備) |
| 3.4.3.4 ベクトル化と検索処理 | [search.py](https://github.com/AvanadeJapanPublishingQuery/AzureOpenAIAgent-RAG-PracticalGuide-book/tree/main/第3章_RAG/3.4%20RAGからAgentへ/3.4.3.4　ベクトル化と検索処理の定義) |
| 3.4.3.5 Agenticな要素 | [agentic.py](https://github.com/AvanadeJapanPublishingQuery/AzureOpenAIAgent-RAG-PracticalGuide-book/tree/main/第3章_RAG/3.4%20RAGからAgentへ/3.4.3.5　Agenticな要素) |
| 3.4.3.6 実行例と出力確認 | [result.py](https://github.com/AvanadeJapanPublishingQuery/AzureOpenAIAgent-RAG-PracticalGuide-book/tree/main/第3章_RAG/3.4%20RAGからAgentへ/3.4.3.6　実行例と出力の確認) |
| 4.1.1 Agent-Based Modeling | [predator_prey.py](https://github.com/AvanadeJapanPublishingQuery/AzureOpenAIAgent-RAG-PracticalGuide-book/tree/main/第4章_AIエージェント/4.1%20エージェントとは/4.1.1%20Agent-Based%20Modeling) |
| 4.3.2 ReActの思考フレームワーク | [llama_index_react.py](https://github.com/AvanadeJapanPublishingQuery/AzureOpenAIAgent-RAG-PracticalGuide-book/tree/main/第4章_AIエージェント/4.3%20エージェントのプロンプト構造と思考フレームワーク/4.3.2%20ReActの思考フレームワーク) |
| 4.4.1 構造化出力機能 | [schedule_agent.py](https://github.com/AvanadeJapanPublishingQuery/AzureOpenAIAgent-RAG-PracticalGuide-book/tree/main/第4章_AIエージェント/4.4%20Azure上でのエージェント技術/4.4.1%20Azure%20OpenAI%20Service%20の構造化出力機能) |
| 4.5.1 AutoGen 基本機能 | [AutoGen 基本](https://github.com/AvanadeJapanPublishingQuery/AzureOpenAIAgent-RAG-PracticalGuide-book/tree/main/第4章_AIエージェント/4.5%20AutoGen概要/4.5.1%20AutoGen%20の基本機能と用途) |
| 4.5.2 AutoGenExtension & LangChainTool | [拡張 / LangChainTool](https://github.com/AvanadeJapanPublishingQuery/AzureOpenAIAgent-RAG-PracticalGuide-book/tree/main/第4章_AIエージェント/4.5%20AutoGen概要/4.5.2%20AutoGenExtensionとLangChainToolの組み合わせ) |
| 4.5.4 Magentic-One 実装 | [Magentic-One](https://github.com/AvanadeJapanPublishingQuery/AzureOpenAIAgent-RAG-PracticalGuide-book/tree/main/第4章_AIエージェント/4.5%20AutoGen概要/4.5.4%20Magentic-One%20によるエージェントシステムの開発) |
| 5.4 簡易なエッジAIシステム構築 | [localslm.py](https://github.com/AvanadeJapanPublishingQuery/AzureOpenAIAgent-RAG-PracticalGuide-book/tree/main/第5章_エッジAI/5.4%20簡易なエッジAIシステム構築) |
| 5.5.1 生成AIモデルの自動最適化 | [app.py](https://github.com/AvanadeJapanPublishingQuery/AzureOpenAIAgent-RAG-PracticalGuide-book/tree/main/第5章_エッジAI/5.5%20MICROSOFT%20OLIVEを使ったエッジAIシステム構築/5.5.1%20生成AIモデルの自動最適化) |

## 9. 免責事項
- 本書は、マイクロソフト主催の「Microsoft Ignite 2024」(2024年11月開催)および「Microsoft Build2025」(2025年5月開催)で、マイクロソフトが発表した内容と、その後のマイクロソフトによる重要なアップデートを可能な限り反映しています。
- 本書は、特に断りのない限り、2025年8月現在の情報に基づいて作成されています。本書で紹介するハードウェア、ソフトウェア、サービスはバージョンアップされる場合があり、ご利用時には、
変更されている場合もありますのであらかじめご了承ください。
- 本書に記載された内容は、情報の提供のみを目的としています。本書の情報の運用の結果について、出版社および著者は一切責任を負わないものとします。
- 本書の内容を参考にされる場合は、必ずお客様自身の責任と判断において、最新の情報と照らし合わせた上での十分な検証をお願いします。
- 本書に記載されている会社名、製品名、サービス名などは、一般に各社の商標または登録商標です。本書では、TM、®、©などのマークを省略しています。
- 本書籍およびサポートサイトで提供するソースコードは、アバナード株式会社の著作物です。
- 本コードは学習・検証目的です。本番利用時はセキュリティ / スケーラビリティ / コスト最適化 / ガバナンスを別途考慮してください。
- 外部 API / サービス仕様変更により動作が変わる可能性があります。

## 10. ライセンス
本リポジトリは MIT License に従います。詳細は `LICENSE` を参照してください。

## 11. 書籍の誤り・エラーについて
- 誤植 / 実行エラー / 改善提案は GitHub Issue にてお願いします。
- Pull Request も歓迎します（再現手順 / 目的 / 影響範囲を明記してください）。
- 正誤表は Issue / Discussions で管理予定です。

## 12. 正誤表
現時点では、正誤表はございません。

## 13. リンク
| 会社名 | リンク | 備考 |
| --- | --- | --- |
| Avanade | https://www.avanade.com/ja-jp | Avanade Japanトップページ |

---

ご活用ありがとうございます。継続的に改善していきます。
