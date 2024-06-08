import os
from typing import Optional

INDEX_BASE = "/Users/johnday/repos/RAGatouille/.ragatouille/colbert/indexes"

def get_index_path(index_name: str) -> Optional[str]:
    path = f"{INDEX_BASE}/{index_name}"
    if os.path.exists(path):
        return path
    print(f"Index '{index_name}' does not exist.")
    return None

def print_search_results(results):
    for i, result in enumerate(results):
        print(f"result {i+1}")
        print(f"score: {result['score']}")
        print(f"rank: {result['rank']}")
        print(f"document_id: {result['document_id']}")
        print(f"passage_id: {result['passage_id']}")
        print(f"content: {result['content']}")
        print("-"*80)
