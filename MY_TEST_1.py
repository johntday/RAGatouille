from multiprocessing import Process, freeze_support
from ragatouille import RAGTrainer
from ragatouille.data import CorpusProcessor, llama_index_sentence_splitter
import random
from ragatouille.utils import get_wikipedia_page

MODEL_NAME = "leaders_colbert"
TOPICS = ["Hayao_Miyazaki", "Studio_Ghibli", "Toei_Animation"]
INDEX_NAME = "leaders"
CHUNK_SIZE = 256
my_full_corpus = []

#
# MAIN
#
def main():
    trainer = RAGTrainer(model_name=MODEL_NAME, pretrained_model_name="colbert-ir/colbertv2.0", language_code="en")

    for topic in TOPICS:
        doc = get_wikipedia_page(topic)
        print(f"topic={topic}, len={len(doc)}, tokens={int(len(doc) / 4)}")
        my_full_corpus.append(doc)

    corpus_processor = CorpusProcessor(document_splitter_fn=llama_index_sentence_splitter)
    documents = corpus_processor.process_corpus(my_full_corpus, chunk_size=CHUNK_SIZE)


    queries = ["What manga did Hayao Miyazaki write?",
               "which film made ghibli famous internationally",
               "who directed Spirited Away?",
               "when was Hikotei Jidai published?",
               "where's studio ghibli based?",
               "where is the ghibli museum?"
    ] * 3
    pairs = []

    for query in queries:
        fake_relevant_docs = random.sample(documents, 10)
        for doc in fake_relevant_docs:
            pairs.append((query, doc))


    trainer.prepare_training_data(raw_data=pairs, data_out_path="./data/", all_documents=my_full_corpus, num_new_negatives=10, mine_hard_negatives=True)


    trainer.train(batch_size=32,
                  nbits=4, # How many bits will the trained model use when compressing indexes
                  maxsteps=500000, # Maximum steps hard stop
                  use_ib_negatives=True, # Use in-batch negative to calculate loss
                  dim=128, # How many dimensions per embedding. 128 is the default and works well.
                  learning_rate=5e-6, # Learning rate, small values ([3e-6,3e-5] work best if the base model is BERT-like, 5e-6 is often the sweet spot)
                  doc_maxlen=256, # Maximum document length. Because of how ColBERT works, smaller chunks (128-256) work very well.
                  use_relu=False, # Disable ReLU -- doesn't improve performance
                  warmup_steps="auto", # Defaults to 10%

                 )

#
# ENDPOINT
#
if __name__ == '__main__':
    freeze_support()
    main()
