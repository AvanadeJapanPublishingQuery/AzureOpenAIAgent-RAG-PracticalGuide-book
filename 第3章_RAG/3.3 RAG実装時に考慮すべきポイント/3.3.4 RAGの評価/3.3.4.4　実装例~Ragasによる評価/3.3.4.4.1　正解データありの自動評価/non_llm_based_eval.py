from ragas import SingleTurnSample
from ragas.metrics import BleuScore

data = {
    "user_input": "与えられたテキストを要約してください\n同社は、アジア市場での好調な業績に牽引され、2024 年第 3 四半期に 8% の上昇を報告しました。この地域での売上は全体の成長に大きく貢献しています。アナリストは、この成功は戦略的なマーケティングと製品のローカリゼーションのおかげであると考えています。アジア市場の好調な傾向は次の四半期も続くと予想されます。",
    "response": "同社は、主に効果的なマーケティング戦略と製品の適応により、2024 年第 3 四半期に 8% の増加を記録し、次の四半期も継続的な成長が見込まれています。",
    "reference": "同社は、戦略的マーケティングとローカライズされた製品によるアジア市場での好調な売上を主因として、2024年第3四半期に8%の成長を報告しており、次の四半期も継続的な成長が見込まれています。"
}
metric = BleuScore()
single_turn = SingleTurnSample(**data)
metric.single_turn_score(single_turn)