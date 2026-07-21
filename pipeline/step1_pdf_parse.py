"""
Step 1: 用 MinerU API 批量解析 PDF
输入: PDF 文件夹
输出: output_dir/{pdf_name}/ 下含 .md + images/
"""
import os, sys, json, argparse, time
from pathlib import Path


def parse_one_pdf(pdf_path, output_dir, api_key, api_url="https://mineru.example.com/api/parse"):
    """调用 MinerU API 解析单个 PDF"""
    import requests

    pdf_name = Path(pdf_path).stem
    out_sub = os.path.join(output_dir, pdf_name)
    os.makedirs(out_sub, exist_ok=True)

    print(f"[INFO] 解析: {pdf_path}")

    with open(pdf_path, "rb") as f:
        resp = requests.post(
            api_url,
            files={"file": f},
            data={"output_dir": out_sub},
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=600,
        )

    if resp.status_code != 200:
        print(f"[ERROR] {pdf_name}: {resp.status_code} {resp.text[:200]}")
        return None

    # MinerU 返回 zip 或直接存到 out_sub
    result = resp.json()
    print(f"[OK] {pdf_name}: {result.get('pages', '?')} pages, "
          f"{result.get('images', '?')} images")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf_dir", required=True, help="PDF 文件夹路径")
    parser.add_argument("--output_dir", default="./parsed", help="解析输出目录")
    parser.add_argument("--api_key", required=True, help="MinerU API Key")
    parser.add_argument("--api_url", default="https://mineru.example.com/api/parse")
    args = parser.parse_args()

    pdf_files = list(Path(args.pdf_dir).glob("*.pdf"))
    print(f"[INFO] 共 {len(pdf_files)} 个 PDF")

    results = {}
    for pdf in pdf_files:
        r = parse_one_pdf(str(pdf), args.output_dir, args.api_key, args.api_url)
        if r:
            results[pdf.stem] = r
        time.sleep(2)  # 限速

    # 保存解析记录
    with open(os.path.join(args.output_dir, "parse_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[DONE] 已解析 {len(results)}/{len(pdf_files)}")


if __name__ == "__main__":
    main()
