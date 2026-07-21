"""
Step 4: 文本增强 + 训练/测试集划分
输入: pairs_refined.json
输出: train_processed.json / test_processed.json
"""
import os, json, random, argparse, time

PARAPHRASE_PROMPT = """请将以下技术手册文本改写成3个不同版本。每个版本保持原意和技术术语不变, 但变换句法结构和措辞。

原文: {original}

请输出3行, 每行一个改写版本, 不加编号和前缀。"""


def generate_paraphrases(text, api_key, api_url, model="gpt-4o-mini"):
    """调用 LLM 生成改写"""
    import requests
    resp = requests.post(
        f"{api_url}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": model,
            "messages": [{"role": "user", "content": PARAPHRASE_PROMPT.format(original=text)}],
            "max_tokens": 500,
            "temperature": 0.8,
        },
        timeout=60,
    )
    content = resp.json()["choices"][0]["message"]["content"].strip()
    lines = [l.strip("- 1234567890. ") for l in content.split("\n") if l.strip()]
    return lines[:3]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="pairs_refined.json")
    parser.add_argument("--output_dir", default="./")
    parser.add_argument("--api_key", default="")
    parser.add_argument("--api_url", default="https://api.openai.com/v1")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--no_augment", action="store_true", help="跳过改写增强")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    with open(args.input, "r", encoding="utf-8") as f:
        pairs = json.load(f)

    # 去重: 按图片路径去重, 保留第一个
    seen = set()
    unique = []
    for p in pairs:
        if p["img"] not in seen:
            seen.add(p["img"])
            unique.append(p)
    print(f"[INFO] {len(pairs)} → {len(unique)} (去重)")

    random.shuffle(unique)
    n_train = int(len(unique) * args.train_ratio)
    train_items = unique[:n_train]
    test_items = unique[n_train:]

    # 文本增强
    if not args.no_augment and args.api_key:
        print(f"[INFO] 为 {len(train_items)} 条训练数据生成改写...")
        for i, item in enumerate(train_items):
            if i % 10 == 0:
                print(f"  {i}/{len(train_items)}")
            try:
                paraphrases = generate_paraphrases(
                    item["original"], args.api_key, args.api_url, args.model)
                item["paraphrases"] = paraphrases
                time.sleep(0.3)
            except Exception as e:
                print(f"  [WARN] 改写失败: {e}")
                item["paraphrases"] = []
    else:
        for item in train_items:
            item["paraphrases"] = []

    for item in test_items:
        item["paraphrases"] = []

    # 保存
    train_out = os.path.join(args.output_dir, "train_processed.json")
    test_out = os.path.join(args.output_dir, "test_processed.json")
    for path, data in [(train_out, train_items), (test_out, test_items)]:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[OK] {path}: {len(data)} 条")

    # 统计
    train_pairs = sum(1 + len(item.get("paraphrases", [])) for item in train_items)
    print(f"\n  训练集: {len(train_items)} 张图, {train_pairs} 个图文对")
    print(f"  测试集: {len(test_items)} 张图")


if __name__ == "__main__":
    main()
