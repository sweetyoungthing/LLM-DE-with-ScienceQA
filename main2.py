import json
import os
import base64
import time
from pathlib import Path
import ollama


def load_problems(json_path):
    """加载问题描述文件"""
    with open(json_path, 'r') as f:
        return json.load(f)


def process_image_question(problem_id, problem_data, image_path):
    """处理单个图片问题"""
    # 构建多模态提示
    with open(image_path, "rb") as img_file:
        base64_image = base64.b64encode(img_file.read()).decode('utf-8')

    prompt = f"""
    请根据图片内容和以下问题进行分析：
    问题：{problem_data['question']}
    选项：{" | ".join(problem_data['choices'])}
    要求：给出选项编号（如0、1、2）和简要解释
    """

    # 调用llava模型（流式响应）
    response = []
    stream = ollama.generate(
        model='llava',
        prompt=prompt,
        images=[base64_image],
        stream=True,
        options={'temperature': 0.2}
    )

    # 收集流式响应
    for chunk in stream:
        if not chunk['done']:
            response.append(chunk['response'])

    return {
        'problem_id': problem_id,
        'question': problem_data['question'],
        'choices': problem_data['choices'],
        'llava_response': "".join(response),
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
    }


def main():
    # 配置路径
    test_dir = Path("test")
    output_file = "llava_responses.json"
    problems = load_problems("problems.json")

    results = []

    # 遍历测试目录
    for folder in test_dir.iterdir():
        if folder.is_dir() and folder.name.isdigit():
            problem_id = folder.name
            problem_data = problems.get(problem_id)

            if problem_data and problem_data.get('image'):
                image_path = folder / problem_data['image']

                if image_path.exists():
                    # 处理单个问题
                    result = process_image_question(problem_id, problem_data, image_path)
                    results.append(result)
                    print(f"已处理问题 {problem_id}")
                else:
                    print(f"警告：图片缺失 {image_path}")

    # 保存结果
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"处理完成，结果已保存至 {output_file}")


if __name__ == "__main__":
    main()