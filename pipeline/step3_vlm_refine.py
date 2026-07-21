"""
Step 3: 用视觉大模型精修图文描述
输入: pairs_raw.json
处理: 对每张图片, 发送给 VLM (Qwen-VL / GPT-4V), 基于原始文本生成干净描述
输出: pairs_refined.json
"""
import os, json, base64, argparse, time
from io import BytesIO
from PIL import Image
from pathlib import Path

PROMPT = """你是一个工业技术文档标注专家。请根据以下技术插图及其原始描述，生成一条准确、简洁的中文描述。

要求:
1. 严格基于图片内容, 不要猜测未显示的信息
2. 保留关键的专业术语 (部件名称、编号、规格参数等)
3. 描述操作步骤或结构关系时使用清晰的顺序逻辑
4. 长度: 20-150 字

原始描述: {raw_text}

请只输出最终描述, 不加任何前缀或解释。"""


def encode_image(image_path):
    """图片 → base64"""
    img = Image.open(image_path).convert("RGB")
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def call_openai_vision(image_path, raw_text, api_key, api_url, model="gpt-4o-mini"):
    """GPT-4V / 兼容 API"""
    import requests
    b64 = encode_image(image_path)
    resp = requests.post(
        f"{api_url}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT.format(raw_text=raw_text)},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                ],
            }],
            "max_tokens": 300,
            "temperature": 0.3,
        },
        timeout=60,
    )
    return resp.json()["choices"][0]["message"]["content"].strip()


def call_qwen_vl(image_path, raw_text, api_key, api_url, model="qwen-vl-max"):
    """Qwen-VL (阿里云 DashScope)"""
    import dashscope
    dashscope.api_key = api_key
    from dashscope import MultiModalConversation

    b64 = encode_image(image_path)
    messages = [{
        "role": "user",
        "content": [
            {"image": f"data:image/jpeg;base64,{b64}"},
            {"text": PROMPT.format(raw_text=raw_text)},
        ],
    }]
    resp = MultiModalConversation.call(model=model, messages=messages)
    return resp.output.choices[0].message.content[0]["text"].strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="pairs_raw.json")
    parser.add_argument("--output", default="pairs_refined.json")
    parser.add_argument("--image_root", required=True, help="图片根目录")
    parser.add_argument("--api_key", required=True)
    parser.add_argument("--api_url", default="https://api.openai.com/v1")
    parser.add_argument("--api_type", default="openai", choices=["openai", "qwen"])
    parser.add_argument("--model", default="gpt-4o-mini")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        pairs = json.load(f)

    results = []
    for i, pair in enumerate(pairs):
        img_path = os.path.join(args.image_root, pair["img"])
        if not os.path.exists(img_path):
            print(f"[SKIP] 图片不存在: {pair['img']}")
            continue

        print(f"[{i+1}/{len(pairs)}] {pair['img']} ...", end=" ", flush=True)
        try:
            if args.api_type == "qwen":
                refined = call_qwen_vl(img_path, pair["raw_text"], args.api_key, args.api_url, args.model)
            else:
                refined = call_openai_vision(img_path, pair["raw_text"], args.api_key, args.api_url, args.model)
            pair["original"] = refined
            results.append(pair)
            print(f"OK ({len(refined)} chars)")
        except Exception as e:
            print(f"FAIL: {e}")
            pair["original"] = pair["raw_text"]  # 降级用原始文本
            results.append(pair)

        time.sleep(0.5)  # 限速

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n[DONE] {len(results)}/{len(pairs)} → {args.output}")


if __name__ == "__main__":
    main()
