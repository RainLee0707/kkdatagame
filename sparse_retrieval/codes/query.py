import json
from tqdm import tqdm
import pandas as pd

def get_query(query_path,result_json):
    loaded_data = pd.read_parquet(query_path)
    columns_to_keep = ['session_id', 'song_id']
    df_loaded_data = loaded_data[columns_to_keep]
    #df_loaded_data = pd.DataFrame(loaded_data)
    df_result_json = pd.DataFrame(result_json)

    merged_df = pd.merge(df_loaded_data, df_result_json, left_on='song_id', right_on='id', how='left')
    merged_df = merged_df.drop(['id', 'song_id'], axis=1)
    updated_data = merged_df.to_dict(orient='records')
    df_updated_data = pd.DataFrame(updated_data)
    grouped_data = df_updated_data.groupby('session_id')['contents'].apply(lambda x: ' '.join(x)).reset_index(name='contents')
    final_result = pd.merge(df_loaded_data[['session_id']], grouped_data, on='session_id', how='left')

    #print(final_result.head(5))
    query = final_result.set_index('session_id')['contents'].to_dict()

    # for session_id, contents in list(query.items())[:5]:
    #     print(f"Session ID: {session_id}, Contents: {contents}")

    return query

# test----------------------------------------------------------------------------

# result_json = []  
# with open('data/collection/collection.jsonl', 'r') as json_file:
#     for line in json_file:
#         entry = json.loads(line)
#         result_json.append(entry)

# file_path = 'data/collection/session_id.jsonl'
# with open(file_path, 'r') as json_file:
#     loaded_data = json.load(json_file)
# query = get_query(loaded_data,result_json)


# test----------------------------------------------------------------------------

 