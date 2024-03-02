from pyserini.pyclass import autoclass
from pyserini.search.lucene import LuceneSearcher
from tqdm import tqdm
import pandas as pd
import os



def search_teacher(searcher, query, args):
    output = open(args.output, 'w')
        
    print(f'Do {args.method} search...')
    for qid, qtext in tqdm(query.items()):
        hits = searcher.search(qtext, k=args.k)
        for i in range(len(hits)):
            # trec format: qid Q0 docid rank score method
            output.write(f'{qid} Q0 {hits[i].docid} {i+1} {hits[i].score:.5f} {args.method}\n')


def search_skip(query_path,searcher, query, args):
    loaded_data = pd.read_parquet(query_path)
    #loaded_data = pd.read_parquet('../datagame-2023/label_train_source.parquet')
    session_song_dict = {}
    print(f'loading data ...')
    for index, row in tqdm(loaded_data.iterrows(), total=len(loaded_data)):
        session_id = row['session_id']
        song_id = row['song_id']
        if session_id in session_song_dict:
            session_song_dict[session_id].append(song_id)
        else:
            session_song_dict[session_id] = [song_id]
    output = open(args.output, 'w')
        
    print(f'Do {args.method} search...')
    for qid, qtext in tqdm(query.items()):
        excluded_song_ids = session_song_dict[qid]
        #print(f"Query {qid}: Excluded Song IDs: {excluded_song_ids}")
        hits = searcher.search(qtext, k=args.k)
        count_satisfied_docs = 0
        for i in range(len(hits)):
            if hits[i].docid not in excluded_song_ids:
                # trec format: qid Q0 docid rank score method
                output.write(f'{qid} Q0 {hits[i].docid} {i+1} {hits[i].score:.5f} {args.method}\n')
                count_satisfied_docs += 1
                # 最多處裡五個output
                if count_satisfied_docs >= 5:
                    break
