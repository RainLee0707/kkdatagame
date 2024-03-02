import pandas as pd

# 讀取Parquet檔案
df = pd.read_parquet('../datagame-2023/label_test_source.parquet')

# 計算DataFrame的行數
total_rows = len(df)

# 切割成10等分
chunk_size = total_rows // 10


chunks = [df.iloc[i * chunk_size: (i + 1) * chunk_size] for i in range(10)]

# 可以將結果存儲為新的Parquet檔案
for i, chunk in enumerate(chunks):
    chunk.to_parquet(f'../data/chunk_{i + 1}.parquet', index=False)
