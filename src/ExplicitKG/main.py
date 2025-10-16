import subprocess
import sys
import os


def run_script(script_name: str):
    """运行指定的Python脚本，并打印输出"""
    try:
        # 正确传递路径，并设置编码为 utf-8
        result = subprocess.run([sys.executable, script_name], check=True, capture_output=True, text=True,
                                encoding='utf-8')
        print(f"成功运行 {script_name}:\n{result.stdout}")
        print(f"错误输出（如果有）：\n{result.stderr}")
    except subprocess.CalledProcessError as e:
        print(f"运行 {script_name} 时发生错误:\n{e.stderr}")
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
        "ExplicitKG/Assembly.py"  # 第四步
        "ExplicitKG/toc_graph.py"
    ]

    # 逐个执行脚本
    for script in scripts:
        print(f"正在执行 {script}...")
        run_script(script)


if __name__ == "__main__":
    main()
