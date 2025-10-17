import subprocess
import sys
import os
from pathlib import Path


def run_script(script_name: str):
    """运行指定的Python脚本，并实时打印输出（含进度条）"""
    try:
        # 关键修改1：合并stdout和stderr，禁用Python子进程的输出缓冲
        env = os.environ.copy()
        env['PYTHONUNBUFFERED'] = '1'  # 禁用输出缓冲，确保进度条实时刷新

        process = subprocess.Popen(
            [sys.executable, script_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # 合并stderr到stdout，进度条会在这里输出
            text=True,
            encoding='utf-8',
            bufsize=0,  # 0 = 无缓冲（比bufsize=1更彻底）
            env=env,  # 传递禁用缓冲的环境变量
            universal_newlines=True
        )

        # 关键修改2：用iter循环读取，提升实时性
        for line in iter(process.stdout.readline, ''):
            print(line, end='', flush=True)  # flush=True 确保主进程即时打印

        # 等待脚本执行完成
        process.wait()

        if process.returncode != 0:
            print(f"运行 {script_name} 时发生错误")
            sys.exit(1)

        print(f"成功运行 {script_name}\n")
    except Exception as e:
        print(f"运行 {script_name} 时发生错误: {e}")
        sys.exit(1)

def main():
    # 设置工作路径（如果需要）
    base_path = r'E:\python\TreeKG\src'
    os.chdir(base_path)

    # 按照要求的顺序依次执行各个文件
    scripts = [
        "ExplicitKG/TextSegmentation.py",  # 第一步
        "ExplicitKG/Summarize.py",  # 第二步
        "ExplicitKG/Extraction.py",  # 第三步
        "ExplicitKG/toc_graph.py"  # 第五步
    ]

    # 逐个执行脚本
    for script in scripts:
        print(f"正在执行 {script}...")
        run_script(script)

if __name__ == "__main__":
    main()
