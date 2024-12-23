import pandas as pd

file_path = 'Heartbeat_TEST.ts'

# 尝试读取文件并指定分隔符
try:
    # 如果文件是制表符分隔（.tsv 文件）
    df = pd.read_csv(file_path, sep='\t')

    # 如果文件是逗号分隔（.csv 文件）
    # df = pd.read_csv(file_path)

    print(df.head())  # 打印前几行
    print(df.info())  # 查看数据的信息
except Exception as e:
    print(f"Error reading the file: {e}")
