from openai import OpenAI
import os
import json
from PIL import Image
import base64
from io import BytesIO
import re  # 新增正则模块

client = OpenAI(
    base_url='http://localhost:11434/v1/',
    api_key='ollama',
)


def encode_image(image_path):
    with Image.open(image_path) as img:
        buffered = BytesIO()
        img.save(buffered, format="png")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')


def process_question(image_path, question_id, question_desc, choices=None):
    base64_image = encode_image(image_path)

    # 强化指令格式（基于网页7的架构特性）
    if choices:
        prompt = f"""Question ID: {question_id}. {question_desc}
Options: {choices}
Respond ONLY with the integer index of the correct option (0-{len(choices) - 1}), formatted as: [答案]X[/答案]"""
    else:
        prompt = f"""Question ID: {question_id}. {question_desc}
Answer the question based on the image. Format your answer as: [答案]ANSWER[/答案]"""

    response = client.chat.completions.create(
        model="llava",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            }
        ],
        temperature=0,  # 固定输出（基于网页8的稳定性建议）
        max_tokens=5  # 限制输出长度（基于网页6的架构限制）
    )

    return response.choices[0].message.content


def main():
    with open('problems.json', 'r') as f:
        problems = json.load(f)

    results = {"experiment_results": []}
    if os.path.exists('results.json'):
        try:
            with open('results.json', 'r') as f:
                results = json.load(f)
        except json.JSONDecodeError:
            results = {"experiment_results": []}

    test_dir = "test"
    for question_id in os.listdir(test_dir):
        question_dir = os.path.join(test_dir, question_id)
        if not os.path.isdir(question_dir):
            continue

        if any(r["question_id"] == question_id for r in results["experiment_results"]):
            continue

        if question_id not in problems:
            continue

        image_path = os.path.join(question_dir, "image.png")
        if not os.path.exists(image_path):
            continue

        question_desc = problems[question_id]["question"]
        choices = problems[question_id].get("choices", None)

        try:
            raw_answer = process_question(image_path, question_id, question_desc, choices)

            # 新增结构化解析逻辑（基于网页7的响应格式）
            answer = None
            if choices:
                match = re.search(r'$$答案$$(\d+)$$/答案$$', raw_answer)
                if match:
                    try:
                        answer_index = int(match.group(1))
                        answer = answer_index if 0 <= answer_index < len(choices) else None
                    except ValueError:
                        pass
            else:
                match = re.search(r'$$答案$$(.+?)$$/答案$$', raw_answer)
                answer = match.group(1).strip() if match else raw_answer

            results["experiment_results"].append({
                "question_id": question_id,
                "question_desc": question_desc,
                "answer": answer,
                "choices": choices if choices else [],
                "correct_answer": problems[question_id].get("answer", None)
            })

            with open('results.json', 'w') as f:
                json.dump(results, f, indent=2)

            print(f"Processed {question_id}: Parsed answer: {answer} | Raw output: {raw_answer}")
        except Exception as e:
            print(f"Error processing {question_id}: {str(e)}")


if __name__ == "__main__":
    main()