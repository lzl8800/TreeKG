import logging
import os
import subprocess
import sys
from pathlib import Path

from log_utils import setup_stage_logger


BASE_PATH = Path(__file__).resolve().parents[1]
LOG_OUTPUT_DIR = BASE_PATH / "output" / "02_hidden_kg"
logger = setup_stage_logger("hidden_main", LOG_OUTPUT_DIR, console_level=logging.INFO)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

SCRIPTS = [
    "HiddenKG/Conv.py",
    "HiddenKG/Aggr.py",
    "HiddenKG/Embedding.py",
    "HiddenKG/Dedup.py",
    "HiddenKG/Pred.py",
    "HiddenKG/FinalKG.py",
]


def run_script(script_rel: str) -> None:
    script_path = (BASE_PATH / script_rel).resolve()
    logger.info("Starting script: %s", script_rel)
    if not script_path.exists():
        logger.error("Script missing: %s", script_path)
        print(f"[ERROR] 脚本不存在：{script_path}")
        sys.exit(1)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    print(f"\n=== 正在执行: {script_rel} ===")
    try:
        proc = subprocess.Popen(
            [sys.executable, str(script_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            cwd=str(BASE_PATH),
        )

        for line in iter(proc.stdout.readline, ""):
            print(line, end="", flush=True)
            logger.info(line.rstrip())

        proc.wait()
        if proc.returncode != 0:
            logger.error("Script failed: %s, returncode=%s", script_rel, proc.returncode)
            print(f"[ERROR] 运行失败：{script_rel}")
            sys.exit(proc.returncode)

        logger.info("Finished script: %s", script_rel)
        print(f"=== 完成: {script_rel} ===\n")

    except KeyboardInterrupt:
        logger.warning("Interrupted while running: %s", script_rel)
        print(f"\n[INTERRUPT] 中断，终止子进程：{script_rel}")
        try:
            proc.terminate()
        except Exception:
            pass
        sys.exit(1)
    except Exception:
        logger.exception("Script raised exception: %s", script_rel)
        print(f"[ERROR] 运行 {script_rel} 时发生异常")
        sys.exit(1)


def main():
    print(f"[INFO] 工作目录：{BASE_PATH}")
    logger.info("HiddenKG pipeline started. cwd=%s, python=%s", BASE_PATH, sys.executable)
    for script in SCRIPTS:
        run_script(script)
    print("\n全部任务完成。")
    logger.info("HiddenKG pipeline finished. log=%s", logger.log_path)


if __name__ == "__main__":
    main()
