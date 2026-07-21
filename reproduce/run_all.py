"""批量重跑实验 (seed=42), 提取 best R@1/5/10 存 results.json"""
import sys, os, subprocess, re, json

SEEDS = ["42", "123", "2024"]

EXPERIMENTS = {
    "qwen_naive":       "Qwen-Naive",
    "mha_pooling":      "Qwen + MHA Pooling",
    "mha_patch":        "Qwen + MHA + Patch",
    "naive_patch":      "Qwen-Naive + Patch",
    "chinese_clip_lora":"Chinese-CLIP (LoRA)",
    "mha_patch_nores":  "Patch w/o residual",
    "qwen_len77":       "Qwen-77",   "qwen_len128": "Qwen-128",
    "qwen_len256":      "Qwen-256",  "qwen_len512": "Qwen-512",
    "mha_heads1":       "MHA heads=1","mha_heads4":"MHA heads=4",
    "mha_heads12":      "MHA heads=12",
    "mha_multiagg":     "MHA 4q",    "mha_q2": "MHA 2q",
    "mha_q8":           "MHA 8q",
}


def extract_metrics(log_text):
    """提取所有评估中 R@1 最高那次对应的 R@1/5/10"""
    matches = re.findall(r"Avg R@1=([\d.]+)%\s*R@5=([\d.]+)%\s*R@10=([\d.]+)%", log_text)
    if not matches:
        m = re.findall(r"best R@1:\s*([\d.]+)%", log_text)
        if m: return {"R@1": max(float(x) for x in m)}
        return None
    # 取 R@1 最高的那次
    best = max(matches, key=lambda x: float(x[0]))
    return {"R@1": float(best[0]), "R@5": float(best[1]), "R@10": float(best[2])}


def run_one(script_name, seed):
    print(f"\n{'='*55}")
    print(f"  {EXPERIMENTS.get(script_name, script_name)}  (seed={seed})")
    print(f"{'='*55}")

    env = os.environ.copy()
    env["SEED"] = seed
    cmd = [sys.executable, f"{script_name}.py"]
    base = os.path.dirname(os.path.abspath(__file__))

    # 实时输出 + 捕获 (只显示 epoch 进度和 eval)
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, env=env, cwd=base, bufsize=1)
    captured = []
    for line in process.stdout:
        captured.append(line)
        line_stripped = line.rstrip()
        if "Epoch" in line_stripped and "|" in line_stripped:
            # tqdm进度条: 只保留最后一行(覆盖打印)
            print(f"\r{line_stripped[:120]}", end='')
        elif "EVAL" in line_stripped or "best" in line_stripped or "完成" in line_stripped:
            print(f"\n{line_stripped}")
    print()  # final newline
    process.wait()
    if process.returncode != 0:
        print(f"[ERROR] 退出码 {process.returncode}")
        return None

    # 清理 checkpoint 释放空间 (保留 mha_patch 最优模型)
    if script_name != "mha_patch":
        for d in os.listdir(base):
            if d.startswith("checkpoints"):
                import shutil
                shutil.rmtree(os.path.join(base, d), ignore_errors=True)

    return extract_metrics(''.join(captured))


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else None
    results = {}

    for script, label in EXPERIMENTS.items():
        if target and script != target:
            continue
        if not os.path.exists(os.path.join(os.path.dirname(__file__), f"{script}.py")):
            print(f"[SKIP] {script}.py 不存在")
            continue

        seeds_runs = {}
        for s in SEEDS:
            m = run_one(script, s)
            if m:
                seeds_runs[s] = m

        if len(seeds_runs) == len(SEEDS):
            import numpy as np
            all_r1 = [seeds_runs[s]["R@1"] for s in SEEDS]
            all_r5 = [seeds_runs[s].get("R@5") for s in SEEDS if seeds_runs[s].get("R@5")]
            all_r10 = [seeds_runs[s].get("R@10") for s in SEEDS if seeds_runs[s].get("R@10")]
            results[label] = {
                "R@1": f"{np.mean(all_r1):.2f} ± {np.std(all_r1):.2f}",
                "R@5": f"{np.mean(all_r5):.2f} ± {np.std(all_r5):.2f}" if all_r5 else "-",
                "R@10": f"{np.mean(all_r10):.2f} ± {np.std(all_r10):.2f}" if all_r10 else "-",
                "seeds": {s: seeds_runs[s] for s in SEEDS},
            }
            print(f"\n  → mean R@1={results[label]['R@1']}")
            # 每完成一个实验就更新 results.json
            out = os.path.join(os.path.dirname(__file__), "results.json")
            with open(out, "w") as f:
                json.dump(results, f, indent=2)

    print(f"\n{'='*50}")
    print("  汇总 (3 seeds, mean ± std)")
    print(f"{'='*50}")
    for label, m in results.items():
        print(f"  {label:35s}  R@1={m['R@1']:>12}  R@5={m.get('R@5','-'):>12}  R@10={m.get('R@10','-'):>12}")

    print(f"\n[INFO] 全部完成")


if __name__ == "__main__":
    main()
