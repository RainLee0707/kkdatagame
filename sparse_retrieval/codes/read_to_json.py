import pandas as pd
import numpy as np
from tqdm import tqdm

meta_song_df = pd.read_parquet('../datagame-2023/meta_song.parquet')
# #看重複的song_id
# duplicates = meta_song_df['song_id'].duplicated()
# duplicate_rows = meta_song_df[duplicates]
# print("Rows with duplicate song_id in meta_song_df:")
# print(duplicate_rows)
meta_song_composer_df = pd.read_parquet('../datagame-2023/meta_song_composer.parquet')
meta_song_genre_df = pd.read_parquet('../datagame-2023/meta_song_genre.parquet')
meta_song_lyricist_df = pd.read_parquet('../datagame-2023/meta_song_lyricist.parquet')
meta_song_producer_df = pd.read_parquet('../datagame-2023/meta_song_producer.parquet')
meta_song_titletext_df = pd.read_parquet('../datagame-2023/meta_song_titletext.parquet')

merged_df = pd.merge(meta_song_df, meta_song_composer_df, on='song_id', how='left')
merged_df = pd.merge(merged_df, meta_song_genre_df, on='song_id', how='left')
merged_df = pd.merge(merged_df, meta_song_lyricist_df, on='song_id', how='left')
merged_df = pd.merge(merged_df, meta_song_producer_df, on='song_id', how='left')
merged_df = pd.merge(merged_df, meta_song_titletext_df, on='song_id', how='left')

train_source_df = pd.read_parquet('../datagame-2023/label_train_source.parquet')
train_target_df = pd.read_parquet('../datagame-2023/label_train_target.parquet')
test_source_df = pd.read_parquet('../datagame-2023/label_test_source.parquet')

train_source_subset = train_source_df[['session_id', 'song_id']]
train_target_subset = train_target_df[['session_id', 'song_id']]
test_source_subset = test_source_df[['session_id', 'song_id']]

# 將 'session_id' 列轉換為字符串類型，再對 train_source_subset 進行 'song_id' 的 groupby，並將對應的 'session_id' 以空格連接
train_source_grouped = train_source_subset.astype({'session_id': str}).groupby('song_id')['session_id'].agg(' '.join).reset_index()
train_target_grouped = train_target_subset.astype({'session_id': str}).groupby('song_id')['session_id'].agg(' '.join).reset_index()
test_source_grouped = test_source_subset.astype({'session_id': str}).groupby('song_id')['session_id'].agg(' '.join).reset_index()

concatenated_data = pd.concat([train_source_grouped, train_target_grouped, test_source_grouped], ignore_index=True)
concatenated_data_grouped = concatenated_data.astype({'session_id': str}).groupby('song_id')['session_id'].agg(' '.join).reset_index()

merged_df = pd.merge(merged_df, concatenated_data_grouped, on='song_id', how='left')


# a = set(merged_df['artist_id'].drop_duplicates())
# b = set(merged_df['song_length'].drop_duplicates())
# c = set(merged_df['album_id'].drop_duplicates())
# d = set(merged_df['album_month'].drop_duplicates())

# print(a, '\n')
# print(b, '\n')
# print(c, '\n')
# print(d, '\n')

genre_mapping = {
    'ce4db56f6a48426643b08038139a8a75': 'Mandarin',
    '6ea61b86b8fff0e3a05bc73ea4eaf21f': 'Western',
    '1f4b914a79eb2bb01c2ea694af626625': 'Hokkien',
    '43244ec4c30a0dad837d982892bc0c05': 'Japanese',
    'bb3d7b04b67d5aeb5ab145bdd70750da': 'Cantonese',
    '5f2a134d2289a8a3de6663ee8e248c8a': 'Soundtrack',
    'bb737ea7450c3abbab1ff613a2c6309e': 'Hip-Hop/Rap',
    'd9ab2ffb929a854e208efaf5297b7cf8': 'Jazz',
    '1d7c8bb87dcc1457ed90240c06f9ebdf': 'Electronic/Dance',
    '7ed5eec2ea6f0208d27f78ef120e52fa': 'Instrumental',
    '157479e86322e5063bce2488bec94d88': 'Classical',
    'ed514a72b48a9d15df7bc4c25eac2c67': 'World',
    '65aeeca3341ca1c6a2ed774aa4e22add': 'Children',
    '74694a488312db91fcd56818fed8b3a6': 'Oldies',
    'a3b51d979053a7f49511b9ab72ee8878': 'Other',
    '5d7d428b81ab3429f189bbc642548b53': 'Games/Comic/Anime',
    '1619db5683958ba19927468990d5ba44': 'Lounge',
    'd36204de09c0c6084d55b1f484a23773': 'Spiritual',
    'ee1696df08c0ac6005e3b9442cdbbded': 'Religious',
    'abdc0a50aff67c591737bc6f57a36e09': 'R&B/Soul',
    '1a619fcf3adfd91711699b7e2cf2c367': 'Spoken Word',
    '2eabe9f164346c7b3ff1bd23078f483e': 'Rock/Alternative',
    'e1d5802bb4e0f6ab79b2b9f4a5ff924b': 'Korean',
    '03c358e326d99a3863e044c5d8e9fb50': 'Blues',
    '4c4ce52d6f6a0e495c407b584ef3e020': 'Malay/Indo',
    '80117354556efbf237f0020bea2f7e42': 'Tamil/Hindi',
    '56203098bef705a44fee44ac0d9b7ef2': 'Thai',
    '09994f2b6bf22721e7379df4e22b1041': 'Country',
    'd48233dd673e7ba524a5a0e3389d39b9': 'Reggae',
    'a2a4a0943fe2c5fe6891d9c34dca906f': 'Alternative',
    'c1d8909eff8a140e7bd0c11508f76dbe': 'Folk',
    'b856b6781d370a3645c6dde0c20b3597': 'Pop',
    '398e2d0befb1c6979e77e5ff7c3fcf07': 'Holiday',
    'b5e874310bf96c3fe29c54a01b982ad7': 'Enka',
    '23ae47469cc0a10de43c453ab59c87b4': 'Christian/Gospel',
    'c3ad773a264597f9f46c2d666e1a8b50': 'Audiobook',
    '5455e1699025b025a3523aba4719e818': 'Audiobook-ChildrensTeenage',
    'f6f02545c1905a8c72c5bf006579996e': 'Audiobook-Education',
}

language_mapping = {
    3: 'Mandarin',
    10: 'Hokkien',
    17: 'Japanese',
    24: 'Cantonese',
    31: 'Korean',
    38: 'Tamil/Hindi',
    45: 'Thai',
    52: 'Western',
    59: 'Malay/Indo',
    60: 'Hakka',
    61: 'Indigenous',
    62: 'English',
    69: 'Vietnamese',
}

merged_df['genre_id'] = merged_df['genre_id'].map(genre_mapping)
merged_df['language_id'] = merged_df['language_id'].map(language_mapping)

selected_columns = [
    'song_id','artist_id','composer_id', 'language_id',
    'genre_id','session_id'
]


# Replace various representations of missing values with numpy.nan
merged_df.replace(['nan', 'None'], np.nan, inplace=True)
# 使用 apply 方法創建 'contents' 欄位
merged_df['contents'] = merged_df[selected_columns].apply(lambda row: ' '.join(str(cell) for cell in row if not pd.isna(cell)), axis=1)
grouped_df = merged_df.groupby('song_id')['contents'].agg(lambda x: ' '.join(x)).reset_index()
result_json = grouped_df[['song_id', 'contents']].to_dict(orient='records')
#print(result_json)

with open('data/collection/collection.jsonl', 'w') as json_file:
    for entry in tqdm(result_json, total=len(result_json)):
        # 寫入格式為 {"id": "song_id", "contents": "39.0 202.0 202.0 Mandarin 2000-01 Mandarin"}
        json_file.write(f'{{"id": "{entry["song_id"]}", "contents": "{entry["contents"]}"}}\n')
