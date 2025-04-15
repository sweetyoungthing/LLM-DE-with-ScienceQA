import numpy as np
import json
# import matplotlib.pyplot as plt
from scipy.optimize import minimize

def irt_difficulty(ability, score):
    """计算IRT难度参数，处理边界情况"""
    alpha = 1.0  # 默认参数
    
    # 处理边界情况
    if score <= 0:
        return 1.0  # 如果得分为0，则认为是最高难度
    elif score >= 1:
        return 0.0  # 如果得分为1，则认为是最低难度
    
    # 正常情况下的计算
    difficulty = ability + np.log((1 / score) - 1) / alpha
    return 1 / (1 + np.exp(-difficulty))  # 归一化到 [0, 1]

def load_model_results(file_path):
    """加载模型结果数据"""
    with open(file_path, 'r') as f:
        return json.load(f)

def calculate_model_abilities(results, target_model):
    """计算指定模型的能力参数"""
    total_problems = len(results)
    correct_count = 0
    
    for problem in results:
        if problem["model_status"][target_model]["correct"]:
            correct_count += 1
    
    # 计算模型能力参数（正确率）
    model_ability = correct_count / total_problems
    return {target_model: model_ability}

def estimate_problem_difficulties(results, model_ability, target_model):
    """基于单个模型估计问题难度"""
    problem_difficulties = {}
    
    for problem in results:
        problem_id = problem["problem_id"]
        # 修改这里：使用模型是否答对作为得分（正确1，错误0）
        score = 1 if problem["model_status"][target_model]["correct"] else 0
        
        difficulty = irt_difficulty(model_ability[target_model], score)
        problem_difficulties[problem_id] = difficulty
    
    return problem_difficulties

def main():
    # 加载模型结果
    results = load_model_results('model_accuracy_results.json')
    
    # 获取用户输入的模型名称
    target_model = input("请输入要分析的模型名称（llava/llama/gemma3/minicpm-v）: ").strip()
    
    # 验证模型有效性
    if target_model not in results[0]["model_status"]:
        raise ValueError(f"无效的模型名称，可用模型: {list(results[0]['model_status'].keys())}")

    # 计算模型能力参数
    model_ability = calculate_model_abilities(results, target_model)
    print(f"\n{target_model} 能力参数: {model_ability[target_model]:.4f}")
    
    # 估计问题难度
    problem_difficulties = estimate_problem_difficulties(results, model_ability, target_model)
    
    # 分析难度分布
    difficulty_categories = analyze_difficulties(problem_difficulties)
    print("\nDifficulty Distribution:")
    for category, count in difficulty_categories.items():
        print(f"Level {category}: {count} problems ({count/len(problem_difficulties)*100:.2f}%)")

    
    # 输出部分问题的难度评级示例
    print("\nDifficulty Rating Examples:")
    sample_size = min(10, len(problem_difficulties))
    sample_problems = list(problem_difficulties.items())[:sample_size]
    for problem_id, difficulty in sample_problems:
        category = categorize_difficulty(difficulty)
        print(f"Problem {problem_id}: Difficulty Value = {difficulty:.4f}, Difficulty Level = {category}")
    
    # 保存难度评级结果
    output = []
    for problem in results:
        problem_id = problem["problem_id"]
        difficulty = problem_difficulties.get(problem_id, 0)
        category = categorize_difficulty(difficulty)
        
        output.append({
            "problem_id": problem_id,
            "accuracy": problem["accuracy"],
            "difficulty_value": difficulty,
            "difficulty_level": category
        })
    
    # 修改输出文件名
    output_file = f'problem_difficulties_{target_model}.json'
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n难度评级已保存至 {output_file}")

def categorize_difficulty(difficulty):
    """将难度值分类为六个等级：1(最简单)到6(最难)"""
    if difficulty < 0.2:
        return 1
    elif difficulty < 0.35:
        return 2
    elif difficulty < 0.5:
        return 3
    elif difficulty < 0.65:
        return 4
    elif difficulty < 0.8:
        return 5
    else:
        return 6

def analyze_difficulties(difficulties):
    """分析难度分布"""
    categories = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
    
    for _, difficulty in difficulties.items():
        category = categorize_difficulty(difficulty)
        categories[category] += 1
    
    return categories

if __name__ == "__main__":
    main()