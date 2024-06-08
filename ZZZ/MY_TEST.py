import os
from ragatouille import RAGPretrainedModel
from ragatouille.my_utils import get_index_path, print_search_results, INDEX_BASE
from ragatouille.utils import get_wikipedia_page

TOPICS = ["Barack_Obama", "Nelson_Mandela", "king"]
INDEX_NAME = "leaders"
my_documents = []
# index_path = get_index_path(INDEX_NAME)
index_path = None

RAG = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")

# CREATE INDEX IF IT DOES NOT EXIST
os.system(f"rm -r {INDEX_BASE}/{INDEX_NAME}")
if index_path is None:
    # get wiki documents for each topic
    for topic in TOPICS:
        doc = get_wikipedia_page(topic)
        print(f"topic={topic}, len={len(doc)}, tokens={int(len(doc)/4)}")
        my_documents.append(doc)

    # create index
    index_path = RAG.index(
        collection=my_documents,
        document_ids=TOPICS,
        # document_metadatas=[{"entity": "person", "source": "wikipedia"}],
        index_name=INDEX_NAME,
        max_document_length=256,
        split_documents=True
    )

    print(f"index_path={index_path}")



# QUERY INDEX
query = "Who was president of the united states of america in 2010 ?"

# retrieve relevant chunks from index
results = RAG.search(query, k=3, index_name=INDEX_NAME)
print_search_results(results)
print("-"*80)

# REUSING INDEX
RAG = RAGPretrainedModel.from_index(index_path)
results = RAG.search(query, k=1, index_name=INDEX_NAME)
print_search_results(results)
print("-"*80)

