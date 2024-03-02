import argparse
import json
from search import LuceneSearcher, search_skip
from query import get_query
from pyserini.pyclass import autoclass

result_json = []
with open('data/collection/collection.jsonl', 'r') as json_file:
    for line in json_file:
        entry = json.loads(line)
        result_json.append(entry)

class BM25Searcher(LuceneSearcher):

    def __init__(self, index):
        super().__init__(index)

    def set_bm25(self, k1=float(0.9), b=float(0.4)):
        BM25Similarity = autoclass('org.apache.lucene.search.similarities.BM25Similarity')(k1, b)
        self.object.searcher = autoclass('org.apache.lucene.search.IndexSearcher')(self.object.reader)
        self.object.searcher.setMaxClauseCount(999999999)
        self.object.searcher.setSimilarity(BM25Similarity)

if __name__ == '__main__':
     for i in range(1, 11):
        parser = argparse.ArgumentParser()
        parser.add_argument("--index", default="indexes/collection", type=str)
        parser.add_argument("--query", default=f'../data/chunk_{i}.parquet', type=str)
        parser.add_argument("--method", default="bm25", type=str)
        parser.add_argument("--k", default=25, type=int)
        parser.add_argument("--output", default=f'runs/bm25_{i}.run', type=str)
        
        args = parser.parse_args()

        if args.method == "bm25":
            searcher = BM25Searcher(args.index)  # Use the new BM25Searcher class
            searcher.set_bm25(k1=0.9, b=0.4)  # Customize BM25 settings

        query = get_query(args.query, result_json)
        search_skip(args.query, searcher, query, args)
        print("第", i, "次")


