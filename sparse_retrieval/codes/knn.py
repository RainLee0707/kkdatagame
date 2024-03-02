import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
from tqdm import tqdm  

train_source_df = pd.read_parquet('../datagame-2023/label_train_source.parquet')

# 將 session_id 和 song_id 轉換為數值表示
le_session = LabelEncoder()
le_song = LabelEncoder()

train_source_df['session_id_encoded'] = le_session.fit_transform(train_source_df['session_id'])
train_source_df['song_id_encoded'] = le_song.fit_transform(train_source_df['song_id'])

# # 創建 KNN 模型
# knn_model = KNeighborsClassifier(n_neighbors=3)

# # 將每個 session 的所有 song_id 放進訓練集
# X_train = train_source_df[['song_id_encoded']]
# y_train = train_source_df['session_id_encoded']

# # 使用 KNN 模型進行訓練
# with tqdm(total=len(X_train), desc="Training KNN model") as pbar:
#     knn_model.fit(X_train, y_train)
#     pbar.update(len(X_train))

# 儲存模型
# joblib.dump(knn_model, 'knn/knn_model.pkl')
knn_model = joblib.load('knn/knn_model.pkl')
test_source_df = pd.read_parquet('../datagame-2023/label_test_source.parquet')
out_df = pd.read_csv('../datagame-2023/output_2.csv')
sample_df = pd.read_csv('../datagame-2023/sample.csv')

# 比較session_id欄位，找出在out.csv中但不在sample.csv中的session_id
selected_session_ids = list(sample_df[~sample_df['session_id'].isin(out_df['session_id'])]['session_id'])
# 選取特定 session_id
#selected_session_ids = [93788, 94029, 97798, 93749, 97669, 97699, 97749, 93679, 93778, 97808, 97939, 708879, 97648, 97858, 97929, 709109, 94089, 708778, 708979, 711609, 93848, 226218, 708699, 709139, 93979, 97909, 708808, 93648, 708969, 93718, 97738, 226239, 708698, 709098, 711558, 94139, 97718, 708988, 709149, 708929, 708999, 93918, 97689, 708819, 709009, 709219, 711729, 709048, 93608, 93709, 97649, 708709, 708839, 708918, 708889, 667098, 93989, 97959, 711608, 708829, 93649, 97908, 708718, 97688, 709099, 709209, 708908, 94069, 97799, 98008, 711589, 3149, 93878, 709058, 709079, 709199, 711549, 708769, 711659, 708749, 93638, 98009, 226238, 709039, 97829, 93689, 94049, 711628, 93999, 93798, 711559, 93839, 708868, 93898, 93998, 94078, 709179, 93708, 94058, 97618, 97989, 708728, 708768, 97768, 709019, 709178, 709189, 94118, 97859, 711618, 93738, 226159, 711569, 711679, 97748, 709049, 708739, 708729, 97958, 711548, 708789, 709088, 711728, 94019, 708779, 708738, 226219, 711519, 93829, 711699, 94138, 708849, 708858, 708928, 97968, 711629, 93799, 93928, 94039, 97679, 93639, 226269, 708788, 709218, 93678, 97878, 709188, 97708, 709078, 94099, 94119, 711658, 93628, 711598, 711709, 94109, 709138, 93849, 709158, 93758, 93948, 93959, 93969, 97659, 97778, 709008, 709148, 711529, 93908, 93808, 94068, 97819, 711528, 97978, 709018, 711568, 94028, 708959, 709029, 711619, 711648, 93968, 97788, 708748, 711698, 93618, 93809, 94038, 97709, 606528, 226278, 93958, 226248, 709069, 226279, 708989, 711578, 694719, 709089, 93819, 97769, 708859, 97728, 711638, 97848, 708919, 708958, 708978, 708948, 97828, 226299, 97839, 709108, 709119, 711649, 93719, 97619, 97999, 708828, 709028, 709169, 711708, 94079, 250729, 708939, 93668, 98018, 708898, 708938, 708968, 709168, 709198, 709208, 93748, 708878, 97628, 711579, 93688, 93978, 94009, 97849, 226289, 708909, 711678, 708848, 708949, 709118, 93609, 711518, 97918, 708809, 708998, 708799, 709059, 709129, 711669, 93698, 94108, 97998, 708899, 93888, 97938, 93889, 711538, 97658, 226249, 709038, 93818, 708798, 711639, 94008, 94129, 97928, 709128, 709159, 711668, 97898, 97919, 93629, 94148, 708838, 226228, 708719, 93739, 93949, 93988, 711539, 93899, 711588, 226298, 94059, 226288, 708708, 708888, 709068, 711599, 2898, 93619, 93869, 226268, 711719, 93699, 94128, 226229, 708818, 708869]
print(len(selected_session_ids))
selected_data_df = test_source_df[test_source_df['session_id'].isin(selected_session_ids)]



# 將 song_id 轉換為數值表示，只考慮訓練資料中見過的 label
selected_data_df['song_id_encoded'] = selected_data_df['song_id'].map(lambda x: le_song.transform([x])[0] if x in le_song.classes_ else -1)

# 移除包含未見過的 label 的資料
selected_data_df = selected_data_df[selected_data_df['song_id_encoded'] != -1]


# 使用先前儲存的 KNN 模型進行預測
predicted_session_ids = knn_model.predict(selected_data_df[['song_id_encoded']])

# 將預測的 session_id 轉換回原始表示
predicted_sessions = le_session.inverse_transform(predicted_session_ids)

# 顯示預測結果
result_df = pd.DataFrame({'original_session_id': selected_data_df['session_id'], 'predicted_session_id': predicted_sessions, 'song_id': selected_data_df['song_id']})
print(result_df)
result_df.to_csv('knn/result_3.csv', index=False)