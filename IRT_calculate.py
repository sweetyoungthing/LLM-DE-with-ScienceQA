import numpy as np

def irt_difficulty(ability, score):
    alpha = 1.0  # 默认参数
    difficulty = ability + np.log((1 / score) - 1) / alpha
    return 1 / (1 + np.exp(-difficulty))  # 归一化到 [0, 1]