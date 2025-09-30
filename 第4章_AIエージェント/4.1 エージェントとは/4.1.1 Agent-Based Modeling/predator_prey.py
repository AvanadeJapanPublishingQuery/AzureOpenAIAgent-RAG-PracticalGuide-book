# 標準ライブラリ
import random
import math

# サードパーティライブラリ
import matplotlib.pyplot as plt


# 乱数の再現性を保つため、シードを固定
random.seed(42)


class Prey:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def move(self):
        # ランダムに(-1,1)の範囲で移動
        self.x += random.uniform(-1, 1)
        self.y += random.uniform(-1, 1)


class Predator:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def move(self):
        # 同じくランダムに移動
        self.x += random.uniform(-1, 1)
        self.y += random.uniform(-1, 1)


def distance(a, b):
    """2次元平面上でのユークリッド距離"""
    return math.hypot(a.x - b.x, a.y - b.y)


def run_simulation(num_preys=15, num_predators=3, steps=100):
    # 初期配置
    preys = [Prey(random.uniform(0, 50), random.uniform(0, 50)) for _ in range(num_preys)]
    predators = [Predator(random.uniform(0, 50), random.uniform(0, 50)) for _ in range(num_predators)]

    # 各ステップのスナップショットを保存する
    snapshots = {}
    for checkpoint in [0, 50, 100]:
        snapshots[checkpoint] = None

    for t in range(steps + 1):
        # エージェントの行動更新：移動
        for p in preys:
            p.move()
        for pd in predators:
            pd.move()

        # 捕食の判定：ある一定距離未満の場合、被食者をリストから除去
        eaten_preys = []
        for pd in predators:
            for p in preys:
                if distance(pd, p) < 1.0:
                    eaten_preys.append(p)

        preys = [p for p in preys if p not in eaten_preys]

        # 指定ステップでスナップショットを記録
        if t in snapshots:
            snapshots[t] = ([(p.x, p.y) for p in preys],
                            [(pd.x, pd.y) for pd in predators])

    return snapshots


def plot_snapshots(snapshots):
    # 縦に3枚のグラフを並べる
    fig, axes = plt.subplots(nrows=3, figsize=(5, 15))
    step_order = sorted(snapshots.keys())

    for i, step in enumerate(step_order):
        preys_pos, preds_pos = snapshots[step]
        ax = axes[i]

        # 被食者（Prey）のプロット
        if preys_pos:
            x_p, y_p = zip(*preys_pos)
            ax.scatter(x_p, y_p, color='green', label='Prey')

        # 捕食者（Predator）のプロット
        if preds_pos:
            x_pd, y_pd = zip(*preds_pos)
            ax.scatter(x_pd, y_pd, color='red', label='Predator')

        ax.set_xlim(0, 50)
        ax.set_ylim(0, 50)
        ax.set_title(f"Step = {step}")
        ax.legend()

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # シミュレーション実行
    snapshots = run_simulation(num_preys=8, num_predators=8, steps=100)
    # 結果プロット
    plot_snapshots(snapshots)
