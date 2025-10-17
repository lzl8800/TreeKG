import subprocess
import sys
import os
from pathlib import Path

# 项目 src 目录
BASE_PATH = Path(r"E:\python\TreeKG\src").resolve()

SCRIPTS = [
    "HiddenKG/Conv.py",      # 第一步
    "HiddenKG/Aggr.py",      # 第二步
    "HiddenKG/Embedding.py", # 第三步
    "HiddenKG/Dedup.py",     # 第四步
    "HiddenKG/Pred.py",      # 第五步
    "HiddenKG/FinalKG.py",   # 第六步
]

def run_script(script_rel: str) -> None:
    """运行单个脚本，实时打印其所有输出（含进度条/日志）"""
    script_path = (BASE_PATH / script_rel).resolve()
    if not script_path.exists():
        print(f"[ERROR] 脚本不存在：{script_path}")
        sys.exit(1)

    env = os.environ.copy()
    # 让子进程不缓冲输出并统一为 UTF-8（进度条可实时刷新）
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    print(f"\n=== 正在执行: {script_rel} ===")
    try:
        proc = subprocess.Popen(
            [sys.executable, str(script_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,   # 合并 stderr，进度条多数走这里
            text=True,
            encoding="utf-8",
            env=env,
            cwd=str(BASE_PATH),         # 不污染主进程的工作目录
        )

        # 逐行转发子进程输出（flush 确保实时）
        for line in iter(proc.stdout.readline, ""):
            print(line, end="", flush=True)

        proc.wait()
        if proc.returncode != 0:
            print(f"[ERROR] 运行失败：{script_rel}")
            sys.exit(proc.returncode)

        print(f"=== 完成: {script_rel} ===\n")

    except KeyboardInterrupt:
        print(f"\n[INTERRUPT] 中断，终止子进程：{script_rel}")
        try:
            proc.terminate()
        except Exception:
            pass
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] 运行 {script_rel} 时发生异常：{e}")
        sys.exit(1)

def main():
    print(f"[INFO] 工作目录：{BASE_PATH}")
    for script in SCRIPTS:
        run_script(script)
    print("\n✅ 全部任务完成。")

if __name__ == "__main__":
    main()
