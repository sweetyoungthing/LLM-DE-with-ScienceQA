import json
import base64
from pathlib import Path
import ollama

def load_problems(json_path):
    """加载问题描述文件"""
    with open(json_path, 'r') as f:
        return json.load(f)

def process_image_question(problem_id, problem_data, image_path):
    """处理单个图片问题"""
    # 读取并编码图片
    with open(image_path, "rb") as img_file:
        base64_image = base64.b64encode(img_file.read()).decode('utf-8')

    # 构建多模态提示
    prompt = f"""
    请根据图片内容和以下问题进行分析,你必须严格遵守我的[要求]。如果没有遵守[要求]，将会面临惩罚：
    [问题]：{problem_data['question']}
    [选项]：{" | ".join(problem_data['choices'])}
    [要求]：给出选项索引（如0、1、2）。（选项索引是从 0 开始的）无论如何，请只给出选项索引，不要任何额外解释,不要加句号。你必须遵守我的要求，否则将会遭到惩罚。
    """

    # 调用llama模型
    response = ollama.generate(
        model='gemma3:4b',
        prompt=prompt,
        images=[base64_image],
        options={'temperature': 0.2}
    )

    return {
        'problem_id': problem_id,
        'question': problem_data['question'],
        'choices': problem_data['choices'],
        'llava_response': response['response'],
        'correct_answer': problem_data['answer']
    }

def main():
    # 配置路径
    json_path = "problems.json"  # JSON文件路径
    test_dir = Path("test")      # 图片目录
    output_file = "result_gemma3.json"  # 输出文件

    # 加载问题和已有结果
    problems = load_problems(json_path)
    try:
        with open(output_file, 'r') as f:
            existing_results = json.load(f)
        processed_ids = {r['problem_id'] for r in existing_results}
    except (FileNotFoundError, json.JSONDecodeError):
        existing_results = []
        processed_ids = set()

    results = existing_results.copy()

    # 遍历处理每个问题
    for problem_id, problem_data in problems.items():
        if problem_id in processed_ids:
            continue
            
        if problem_data.get('image'):
            image_path = test_dir / problem_id / problem_data['image']
            
            if image_path.exists():
                # 处理问题
                result = process_image_question(problem_id, problem_data, image_path)
                results.append(result)
                print(f"已处理问题 {problem_id}")
                
                # 立即保存当前结果
                with open(output_file, 'w') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
            else:
                print(f"无图片： {image_path}")

    print(f"处理完成，结果已保存至 {output_file}")

if __name__ == "__main__":
    main()