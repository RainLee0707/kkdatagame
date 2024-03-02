import csv

# 讀取run檔數據
file_path = 'runs/bm25.run'  # 請更換為實際的文件路徑
with open(file_path, 'r') as file:
    run_data = file.read()

# 初始化一個字典來存儲結果
result_dict = {}

# 將run檔的每一行解析為字典
lines = run_data.strip().split('\n')
for line in lines:
    parts = line.split()
    session_id = parts[0]
    top_value = parts[2]
    result_dict.setdefault(session_id, []).append(top_value)

# 寫入CSV文件
csv_filename = '../datagame-2023/output.csv'
with open(csv_filename, 'w', newline='') as csvfile:
    fieldnames = ['session_id', 'top1', 'top2', 'top3', 'top4', 'top5']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    # 寫入CSV標題行
    writer.writeheader()

    # 寫入CSV數據行
    for session_id, top_values in result_dict.items():
        row_data = {'session_id': session_id}
        for i, top_value in enumerate(top_values[:5], start=1):
            row_data[f'top{i}'] = top_value
        writer.writerow(row_data)

print(f'CSV檔案已生成：{csv_filename}')
