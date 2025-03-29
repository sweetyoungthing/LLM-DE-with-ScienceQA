from openai import OpenAI
import os
import json
from PIL import Image
import base64
from io import BytesIO

# 创建 OpenAI 客户端
client = OpenAI(
    base_url='http://localhost:11434/v1/',
    api_key='ollama',
)

def encode_image(image_path):
    with Image.open(image_path) as img:
        buffered = BytesIO()
        img.save(buffered, format="png")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

def process_question(image_path, question_id, question_desc):
    # 编码图片
    base64_image = encode_image(image_path)
    
    # 构造问题
    question = f"Question ID: {question_id}. {question_desc} Please answer this question based on the image."
    
    # 调用模型
    response = client.chat.completions.create(
        model="llava",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            }
        ],
    )
    
    return response.choices[0].message.content

def main():
    # 加载 problems.json
    with open('problems.json', 'r') as f:
        problems = json.load(f)
    
    # 加载已有结果
    results = {"experiment_results": []}
    if os.path.exists('results.json'):
        try:
            with open('results.json', 'r') as f:
                results = json.load(f)
        except json.JSONDecodeError:
            print("Warning: results.json is empty or invalid. Starting with an empty results file.")
            results = {"experiment_results": []}
    
    # 遍历 test 文件夹
    test_dir = "test"
    for question_id in os.listdir(test_dir):
        question_dir = os.path.join(test_dir, question_id)
        if not os.path.isdir(question_dir):
            continue
            
        # 检查是否已处理过
        processed = any(r["question_id"] == question_id for r in results["experiment_results"])
        if processed:
            continue
            
        # 检查是否在 problems.json 中
        if question_id not in problems:
            print(f"Question {question_id} not found in problems.json, skipping...")
            continue
            
        # 获取图片路径
        image_path = os.path.join(question_dir, "image.png")
        if not os.path.exists(image_path):
            print(f"Image not found for question {question_id}, skipping...")
            continue
            
        # 获取问题描述
        question_desc = problems[question_id]["question"]
        
        # 处理问题
        try:
            answer = process_question(image_path, question_id, question_desc)
            results["experiment_results"].append({
                "question_id": question_id,
                "question_desc": question_desc,
                "answer": answer,
                "choices": problems[question_id].get("choices", []),
                "correct_answer": problems[question_id].get("answer", None)
            })
            
            # 保存结果
            with open('results.json', 'w') as f:
                json.dump(results, f, indent=2)
                
            print(f"Processed question {question_id}")
        except Exception as e:
            print(f"Error processing question {question_id}: {str(e)}")

if __name__ == "__main__":
    main()