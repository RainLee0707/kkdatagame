import pandas as pd


file_path = "knn/result_2.csv"
df = pd.read_csv(file_path)


most_frequent_predictions = df.groupby('original_session_id')['predicted_session_id'].agg(lambda x: x.value_counts().idxmax()).reset_index()


# print(most_frequent_predictions)
most_frequent_predictions.to_csv("knn/session_2.csv", index=False)


most_frequent_predictions = pd.read_csv("knn/session_2.csv")
train_target_df = pd.read_parquet('../datagame-2023/label_train_target.parquet')
train_target_subset = train_target_df[['session_id', 'song_id']]
train_target_grouped = train_target_subset.groupby('session_id')['song_id'].apply(lambda x: x.head(5).tolist()).reset_index()


merged_df = pd.merge(most_frequent_predictions, train_target_grouped, how='left', left_on='predicted_session_id', right_on='session_id')
merged_df = merged_df.rename(columns={'song_id': 'top'})

merged_df[['top1', 'top2', 'top3', 'top4', 'top5']] = pd.DataFrame(merged_df['top'].tolist(), index=merged_df.index)

result_df = merged_df[['original_session_id', 'top1', 'top2', 'top3', 'top4', 'top5']]
result_df.to_csv("knn/result_2_session.csv", index=False)


print("Results saved to result_session.csv")