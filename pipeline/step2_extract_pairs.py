"""
Step 2: 从 MinerU 解析的 Markdown 中提取图文对
策略: 匹配 md 中的图片引用和最近的上下文文本
输出: pairs.json (每项含 img_path, raw_text, context)
"""
import os, re, json, argparse
from pathlib import Path


def extract_pairs(md_path, img_dir):
    """从单个 md 文件中提取图文对"""
    with open(md_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    pairs = []
    img_pattern = re.compile(r"!\[.*?\]\((.+?)\)")

    for i, line in enumerate(lines):
        m = img_pattern.search(line)
        if not m:
            continue

        img_name = m.group(1)
        img_path = os.path.join(img_dir, img_name)
        if not os.path.exists(img_path):
            # 尝试查找实际位置的图片
            found = list(Path(img_dir).glob(f"**/{img_name}"))
            if found:
                img_path = str(found[0])
            else:
                continue

        # 提取上下文: 图片前后各3行, 排除空行和纯符号行
        start = max(0, i - 3)
        end = min(len(lines), i + 4)
        context_lines = []
        for j in range(start, end):
            if j == i:
                continue
            stripped = lines[j].strip()
            if stripped and not stripped.startswith("#") and len(stripped) > 3:
                context_lines.append(stripped)

        context = " ".join(context_lines).strip()
        if not context or len(context) < 5:
            continue

        # 用上下文作为初始文本描述
        raw_text = lines[i - 1].strip() if i > 0 else ""
        if not raw_text or raw_text.startswith("!") or raw_text.startswith("#"):
            raw_text = context

        pairs.append({
            "img": os.path.relpath(img_path, os.path.dirname(os.path.dirname(img_dir))),
            "raw_text": raw_text[:500],
            "context": context[:500],
        })

    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parsed_dir", required=True, help="MinerU 解析输出目录 (含各 PDF 子目录)")
    parser.add_argument("--output", default="pairs_raw.json")
    args = parser.parse_args()

    all_pairs = []
    total = 0
    for sub in Path(args.parsed_dir).iterdir():
        if not sub.is_dir():
            continue
        md_files = list(sub.glob("*.md"))
        img_dir = os.path.join(str(sub), "images")
        for md_file in md_files:
            pairs = extract_pairs(str(md_file), img_dir)
            if pairs:
                print(f"[OK] {sub.name}/{md_file.name}: {len(pairs)} 对")
            all_pairs.extend(pairs)
            total += len(pairs)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_pairs, f, ensure_ascii=False, indent=2)
    print(f"\n[DONE] 共 {len(all_pairs)} 个图文对 → {args.output}")


if __name__ == "__main__":
    main()
