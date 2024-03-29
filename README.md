# KKCompany Music Challenge: Next-5 Songcraft

## 概述

本專案旨在透過原始資料，建立模型並提供音樂推薦服務。以下是詳細的步驟和說明。

## 系統要求

確保您的系統已安裝WSL並切換至Linux系統。pyserini 套件目前只支援linux系統 請使用linux環境執行。

## 資料準備

1. 請至 [KKCompany Music Challenge: Next-5 Songcraft](https://www.kaggle.com/competitions/datagame-2023) 下載原始資料，並將其放入 `datagame-2023` 資料夾中。

2. 使用 `code/read_to_json.py` 將資料轉換成 JSON 格式，以便進行後續的索引建構。

3. 執行 `build_index.sh` 腳本以建立索引。這將利用原始資料中的 `song_id`、`artist_id`、`composer_id`、`language_id`、`genre_id` 和 `session_id` 等欄位建立初期向量。

## 模型建構

語言向量模型的應用是為了進行音樂推薦服務。透過建立一個語言向量模型，我們可以將原始音樂數據轉換成數學表示，以便計算歌曲之間的相似度。以下是語言向量模型在這個專案中的詳細應用：

1. **轉換原始數據成向量表示：** 我們利用語言向量模型將原始數據中的 `song_id`、`artist_id`、`composer_id`、`language_id`、`genre_id` 和 `session_id` 等欄位透過PYSERINI的INDEX建構模型方式，將這些欄位轉換為向量表示。這些向量可以捕捉到歌曲之間的相似性和關聯性。

2. **建立初期向量：** 透過將這些欄位轉換為向量表示，我們建立了一組初期向量。這些向量包含了原始數據中的信息，例如歌曲的特徵、音樂家的風格等等。

3. **計算歌曲相似度：** 利用這些向量表示，我們可以使用不同的相似度計算方法來衡量歌曲之間的相似度。在這個專案中，我們使用了 BM25 方法來計算歌曲之間的相似度，從而找出與某一首歌曲相似的其他歌曲。

4. **音樂推薦服務：** 通過計算歌曲相似度，我們可以為用戶提供個性化的音樂推薦服務。根據用戶的喜好和聽歌習慣，我們可以推薦與他們喜歡的歌曲相似的其他歌曲，從而提升用戶的音樂體驗。

## 使用步驟

1. 將原始資料放入 `datagame-2023` 資料夾中。

2. 使用 `code/read_to_json.py` 將資料轉換為 JSON 格式。

3. 執行 `build_index.sh` 腳本以建立索引。

4. 執行 `code/main.py` 以建立模型。如果檔案過大無法處理，可以先執行 `cut_file.py` 將其分割並存入 `../data` 目錄中。

5. 處理部分遺失資料。由於某些 session 可能無法透過語言模型進行，請注意處理這些缺失資料。

## 處理缺失值

每個 session 被視為一個個人用戶，並且可以根據 `label_test_source.parquet` 和 `label_train_target.parquet` 中的資料推斷前20首歌曲會有那些推薦歌單，從而使用 KNN 方法找出相同的用戶歌單進而推薦缺失資料。
