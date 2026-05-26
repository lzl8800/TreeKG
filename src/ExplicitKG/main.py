import logging
import os
import subprocess
import sys
from pathlib import Path

from log_utils import setup_stage_logger


BASE_PATH = Path(__file__).resolve().parents[1]
LOG_OUTPUT_DIR = BASE_PATH / "output" / "01_explicit_kg"
logger = setup_stage_logger("explicit_main", LOG_OUTPUT_DIR, console_level=logging.INFO)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def run_script(script_name: str):
    try:
        logger.info("Starting script: %s", script_name)
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"

        process = subprocess.Popen(
            [sys.executable, str(BASE_PATH / script_name)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=0,
            env=env,
            universal_newlines=True,
            cwd=str(BASE_PATH),
        )

        for line in iter(process.stdout.readline, ""):
            print(line, end="", flush=True)
            logger.info(line.rstrip())

        process.wait()
        if process.returncode != 0:
            logger.error("Script failed: %s, returncode=%s", script_name, process.returncode)
            print(f"运行 {script_name} 时发生错误")
            sys.exit(1)

        logger.info("Finished script: %s", script_name)
        print(f"成功运行 {script_name}\n")
    except Exception:
        logger.exception("Script raised exception: %s", script_name)
        print(f"运行 {script_name} 时发生错误")
        sys.exit(1)


def main():
    scripts = [
        "ExplicitKG/TextSegmentation.py",
        "ExplicitKG/Summarize.py",
        "ExplicitKG/Extraction.py",
        "ExplicitKG/toc_graph.py",
    ]

    print(f"工作目录：{BASE_PATH}")
    logger.info("ExplicitKG pipeline started. cwd=%s, python=%s", BASE_PATH, sys.executable)
    for script in scripts:
        print(f"正在执行 {script}...")
        run_script(script)
    logger.info("ExplicitKG pipeline finished. log=%s", logger.log_path)


if __name__ == "__main__":
    main()
