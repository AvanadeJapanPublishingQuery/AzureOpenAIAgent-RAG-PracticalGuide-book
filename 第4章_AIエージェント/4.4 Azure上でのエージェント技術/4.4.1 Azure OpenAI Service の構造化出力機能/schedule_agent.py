import os
import json
import asyncio
import random
import httpx
import pandas as pd
import matplotlib.pyplot as plt
from openai import AsyncAzureOpenAI as AzureOpenAI

# ---------------------
# OpenAIClientクラス
# ---------------------
class OpenAIClient:
    api_configurations = [
        {"endpoint": os.environ.get("OPENAI_API_BASE_URL", ""), "api_key": os.environ.get("OPENAI_API_KEY", "")},
    ]
    api_version: str = os.environ.get("OPENAI_API_VERSION", "2023-12-01-preview")
    engine_name_llm: str = os.environ.get("OPENAI_API_GPT4_OMNI_128K_20240806", "")
    http_client: httpx.Client = httpx.AsyncClient()
    max_retries: int = int(os.environ.get("MAX_RETRIES", 3))
    
    @classmethod
    async def get_openai_client(cls, api_key: str, endpoint: str) -> AzureOpenAI:
        """Class method to get or create an AzureOpenAI client with class-level configuration."""
        return AzureOpenAI(
            api_key=api_key,
            api_version=cls.api_version,
            azure_endpoint=endpoint,
            http_client=cls.http_client,
            timeout=httpx.Timeout(
                connect=float(os.environ.get("CONNECT_TIMEOUT", 2.0)),
                read=float(os.environ.get("READ_TIMEOUT", 20.0)),
                write=None,
                pool=None
            ),
        )

    async def get_client_with_retries(self) -> AzureOpenAI:
        """Tries to get an OpenAI client using different configurations with retry logic."""
        # 用意したapi_configurationsリストをランダムシャッフル
        configurations = self.api_configurations
        random.shuffle(configurations)

        for attempt in range(self.max_retries):
            config = configurations[attempt % len(configurations)]
            try:
                # 現在のconfigでクライアントを取得する
                client = await self.get_openai_client(config['api_key'], config['endpoint'])
                print(f"Successfully obtained client with endpoint: {config['endpoint']}")
                return client
            except Exception as e:
                print(f"Error while getting client for {config['endpoint']}: {e}")
                if attempt == self.max_retries - 1:
                    raise Exception("Max retries exceeded. Could not obtain a valid OpenAI client.")
        raise Exception("Failed to get OpenAI client after retries.")

    # GPT API interaction logic
    async def gpt_call(self, client, prompt: str, text: str) -> str:
        # ここでresponse_format={"type": "json_object"}を指定してJsonModeを有効化
        response = await client.chat.completions.create(
            model=self.engine_name_llm,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": text}
            ]   
        )
        # JSONとして返るが、ここでは文字列として返却している
        return str(response.choices[0].message.content)

# ---------------------
# Agent A: ScheduleGenerator
# ---------------------
class ScheduleGenerator:
    def __init__(self, client: OpenAIClient, constraints, employee_info):
        self.client = client
        self.constraints = constraints
        self.employee_info = employee_info

    async def generate_schedule(self, text):
        client = await self.client.get_client_with_retries()  # 有効なclientを取得
        # Promptにスケジュール生成の意図とフォーマットを明示
        prompt = (
            "以下の社員情報および制約条件に基づいて、一週間の勤務スケジュールを作成してください。\n"
            f"【社員情報＆希望勤務形態】:\n{self.employee_info}\n"
            f"【制約条件】:\n{self.constraints}\n"
            "出力形式（JSON）は以下の通りとしてください。:\n"
            "{\n"
            "  '日付': ['月曜日 YYYY-MM-DD', '火曜日YYYY-MM-DD'...],\n"
            "  '山田太郎': ['日勤', '夜勤', '休日' ...],\n"
            "  '佐藤花子': ['夜勤', '日勤', '日勤', ...],\n"
            "  '鈴木一郎': ['休日', '日勤', '夜勤', ...],\n"
            "}\n"
        )
        # Agent Bからのフィードバックなどを textに含めて呼び出す
        schedule = await self.client.gpt_call(client, prompt, text)
        return schedule

# ---------------------
# Agent B: ScheduleEvaluator
# ---------------------
class ScheduleEvaluator:
    def __init__(self, client: OpenAIClient, constraints, employee_info):
        self.client = client
        self.constraints = constraints
        self.employee_info = employee_info

    async def evaluate_schedule(self, schedule):
        client = await self.client.get_client_with_retries()
        # スケジュールを評価するためのプロンプト
        prompt = (
            f"以下の社員情報および制約条件に基づいて、提示された勤務スケジュールの妥当性を評価してください。:\n"
            f"【社員情報】:\n{self.employee_info}\n"
            f"【制約条件】:\n{self.constraints}\n"
            f"【スケジュール】:\n{schedule}\n"
            "以下の形式で評価結果を返してください。\n"
            "{\n"
            "  'score': 0-100,  // スケジュールが制約をどれだけ満たしているかのスコア\n"
            "  'feedback': 'string'  // 改善のためのフィードバック\n"
            "}"
        )
        evaluation = await self.client.gpt_call(client, prompt, "")
        # 文字列として受け取り、json.loads()でPythonの辞書へ変換
        return json.loads(evaluation)

# ---------------------
# Main system: ScheduleOptimizer
# ---------------------
class ScheduleOptimizer:
    def __init__(self, client, constraints, employee_info):
        self.client = client
        self.constraints = constraints
        self.employee_info = employee_info
        self.schedule_generator = ScheduleGenerator(client, constraints, employee_info)
        self.schedule_evaluator = ScheduleEvaluator(client, constraints, employee_info)
        self.schedule_file = "schedule_data.json"

    def load_schedule(self):
        """Load the existing schedule from a JSON file if it exists."""
        if os.path.exists(self.schedule_file):
            with open(self.schedule_file, 'r') as f:
                return json.load(f)
        return None

    def save_schedule(self, schedule):
        """Save the schedule to a JSON file."""
        with open(self.schedule_file, 'w') as f:
            json.dump(schedule, f, indent=4)

    async def optimize_schedule(self, initial_text: str, expected_score: int = 90, max_iterations=5):
        # 既存のスケジュールがあればロードして初期テキストを更新
        existing_schedule = self.load_schedule()
        if existing_schedule:
            print("Existing schedule found. Using as seed for optimization.")
            initial_text = json.dumps(existing_schedule)
        else:
            print("No existing schedule found. Generating a new one.")

        feedback_list = []  # フィードバックの履歴
        best_score = -1
        best_schedule = None

        for iteration in range(max_iterations):
            print(f"Iteration {iteration + 1}:")
            # Agent Aでスケジュールを生成
            schedule = await self.schedule_generator.generate_schedule(initial_text)
            print(f"Generated Schedule:\n{schedule}\n")

            # Agent Bで評価
            evaluation = await self.schedule_evaluator.evaluate_schedule(schedule)
            print(f"Evaluation and Suggestions:\n{evaluation['feedback']}\n")

            score = evaluation.get('score', 0)
            print(f"Score: {score}")

            if score > best_score:
                best_score = score
                best_schedule = schedule
                print("New best schedule found with higher score.")

            if score >= expected_score:
                print("Optimized schedule found.")
                self.save_schedule(best_schedule)
                break
            else:
                print("Revising the schedule based on feedback...\n")
                feedback_list.append(evaluation['feedback'])
                if len(feedback_list) > 2:
                    feedback_list.pop(0)
                
                initial_text = "\n".join(feedback_list)

        if best_schedule:
            self.save_schedule(best_schedule)
            self.visualize_schedule(best_schedule)
        return best_schedule

    def visualize_schedule(self, schedule: str):
        """Convert the final schedule (in JSON string format) to a DataFrame and plot it."""
        try:
            schedule_dict = json.loads(schedule)
            df = pd.DataFrame(schedule_dict)
            plot_schedule(df)
        except ValueError as e:
            print(f"Error parsing the schedule JSON: {e}")

# ---------------------
# スケジュール可視化用の関数
# ---------------------
def plot_schedule(df):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('tight')
    ax.axis('off')

    table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)

    # 各シフトによって色分けを行う例
    for i in range(len(df)):
        for j in range(1, len(df.columns)):
            if df.iloc[i, j] == 'Off':
                table[(i+1, j)].set_facecolor('#FFFF99')
            elif df.iloc[i, j] == 'Night':
                table[(i+1, j)].set_facecolor('#556B2F')
            elif df.iloc[i, j] == 'Afternoon':
                table[(i+1, j)].set_facecolor('#C0C0C0')
            elif df.iloc[i, j] == 'Morning':
                table[(i+1, j)].set_facecolor('#87CEEB')

    plt.title('Employee Schedule', fontsize=16)
    plt.show()

# ---------------------
# スケジュールの制約条件
# ---------------------
constraints = """
- 各日ごとの勤務人数上限:日勤は最大2名、夜勤は1名まで
- 全員に週2回の休日を与えること
- 夜勤→日勤の連続は避けること(最低1日空ける)

"""

# ---------------------
# 従業員情報
# ---------------------
employee_info = """
【社員情報&希望勤務形態】Employee Information:
- 山田 太郎(ID: E001):日勤中心、週休2日
- 佐藤 花子(ID: E002):夜勤可、連続勤務は最大3日まで
- 鈴木 一郎(ID: E003):水曜日は必ず休日、日曜は勤務可能

"""

# ---------------------
# 実行部分
# ---------------------
client = OpenAIClient()
optimizer = ScheduleOptimizer(client, constraints, employee_info)

# イベントループを起動し、最適化を実行
optimized_schedule = asyncio.run(optimizer.optimize_schedule("Initial schedule request"))
print(f"Final Optimized Schedule:\n{optimized_schedule}")
