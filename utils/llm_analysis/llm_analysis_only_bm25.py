from fastbm25 import fastbm25
from typing import List, Dict
import pandas as pd
from tqdm import tqdm
import random


# all_documents_file_path = "C:\MY_FILES\PycharmProjects\CiteBART\preprocessing\global_datasets\\acl200_global\context_dataset.csv"

eval_set_for_masked_contexts_file_path = "C:\MY_FILES\PycharmProjects\CiteBART\preprocessing\global_datasets\\arxiv_global\context_dataset_eval.csv"

train_set_file_path = "C:\MY_FILES\PycharmProjects\CiteBART\preprocessing\global_datasets\\arxiv_global\context_dataset_train.csv"

# Example document format:
# documents = [
#     {"citation": "author-name-year-1", "title": "Deep Learning for NLP", "abstract": "This paper explores..."},
#     {"citation": "author-name-year-2", "title": "BERT: Pre-training...", "abstract": "We introduce a new model..."},
#     ...
# ]

class CitationRetrieverBM25:
    def __init__(self, documents: List[Dict[str, str]]):
        # Combine author-date citation, title, and abstract as the document text
        self.doc_texts = [
            f"{doc['citation']} [SEP] {doc['title']} [SEP] {doc['abstract']}" for doc in documents
        ]

        # Store original metadata
        self.documents = documents

        # Tokenize each document for BM25
        tokenized_corpus = [doc.lower().split(" ") for doc in self.doc_texts]
        # tokenized_corpus = [word_tokenize(text.lower()) for text in self.doc_texts]

        self.bm25 = fastbm25(tokenized_corpus)

    def retrieve_top_k(self, masked_context: str, k: int = 100) -> List[Dict[str, str]]:
        # Tokenize the masked context
        # query_tokens = word_tokenize(masked_context.lower())
        query_tokens = masked_context.lower().split()

        # Get top k indices
        top_indices = self.bm25.top_k_sentence(query_tokens, k=k)  # Returns List of [(nearest sentence,index,score)]

        #  print(top_indices, "\n\n")

        # Return top k documents with their metadata
        return [self.documents[idx[1]] for idx in top_indices]


"""     ### EXAMPLE USAGE ###
docs = [{"citation": "Vinyals et al., 2014", "title": "Grammar as a Foreign Language", "abstract": "Syntactic parsing is a fundamental problem in computational linguistics and Natural Language Processing. Traditional approaches to parsing are highly complex and problem specific. Recently, Sutskever et al. (2014) presented a domain-independent method for learning to map input sequences to output sequences that achieved strong results on a large scale machine translation problem. In this work, we show that precisely the same sequence-to-sequence method achieves results that are close to state-of-the-art on syntactic constituency parsing, whilst making almost no assumptions about the structure of the problem."},
        {"citation": "Sutskever et al., 2014", "title": "Sequence to Sequence Learning with Neural Networks", "abstract": "Deep Neural Networks (DNNs) are powerful models that have achieved excellent performance on difficult learning tasks. Although DNNs work well whenever large labeled training sets are available, they cannot be used to map sequences to sequences. In this paper, we present a general end-to-end approach to sequence learning that makes minimal assumptions on the sequence structure. Our method uses a multilayered Long Short-Term Memory (LSTM) to map the input sequence to a vector of a fixed dimensionality, and then another deep LSTM to decode the target sequence from the vector. Our main result is that on an English to French translation task from the WMT-14 dataset, the translations produced by the LSTM achieve a BLEU score of 34.7 on the entire test set, where the LSTM's BLEU score was penalized on out-of-vocabulary words. Additionally, the LSTM did not have difficulty on long sentences. For"}]
masked_citing_context = "nd-crafted features, lexicons, and grammars.Meanwhile, recurrent neural networks  have made swift inroads intomany structured prediction tasks in NLP,including machine translation andsyntactic parsing  <mask>  .Because RNNs make very few domain-specific assumptions,they have the potential to succeed at a wide variety of taskswith minimal feature engineering.wever, this flexibility also puts RNNs at a disadv"
retriever = CitationRetrieverBM25(docs)
top_k_docs = retriever.retrieve_top_k(masked_citing_context, k=1)
print(top_k_docs)
"""

# Open train set documents csv file and read the documents
df_train = pd.read_csv(train_set_file_path)
# Convert to list of dictionaries. From df, the column names are "masked_token_target", "target_title", "target_abstract"
temp_train_docs = df_train[["masked_token_target", "target_title", "target_abstract"]].to_dict(orient='records')

# Open eval set documents csv file and read the documents.
# Convert to list of dictionaries. From df, the column names are "masked_token_target", "target_title", "target_abstract"
df_eval = pd.read_csv(eval_set_for_masked_contexts_file_path)
temp_eval_docs = df_eval[["masked_token_target", "target_title", "target_abstract"]].to_dict(orient='records')

# Merge the train and eval set documents
temp_docs = temp_train_docs + temp_eval_docs


# The new column names should be "citation", "title", "abstract", respectively. Rename them.
docs = [
    {
        "citation": doc["masked_token_target"],
        "title": doc["target_title"],
        "abstract": doc["target_abstract"]
    }
    for doc in temp_docs
]

# Make sure every entry in the docs is unique. Remove the entire entries with the same citation field.
unique_docs = {doc['citation']: doc for doc in docs}.values()
docs = list(unique_docs)

print("\nTotal number of unique papers from the entire dataset:", len(docs), "\n")

df_eval = pd.read_csv(eval_set_for_masked_contexts_file_path)

temp_eval_set_masked_contexts = df_eval[["masked_cit_context", "citing_title", "citing_abstract"]].to_dict(orient='records')

# These are the ground truth citations for the evaluation set. After LLM model prediction, we will compare the predicted citations with these ground truth citations.
ground_truth_citations = df_eval["masked_token_target"].tolist()

eval_set_masked_contexts = [
    f"{d['masked_cit_context']} [SEP] {d['citing_title']} [SEP] {d['citing_abstract']}" for d in temp_eval_set_masked_contexts
]


# -------------------------- Reduce the number of documents in "eval set" to 30000 by sampling randomly  -------------------------------
# Set a seed for reproducibility
random.seed(42)

# Sample 30000 documents from the eval set
if len(eval_set_masked_contexts) > 30000:
    # Get 30000 random indices between 0 and the length of eval_set_masked_contexts
    random_indices = random.sample(range(len(eval_set_masked_contexts)), 30000)
    # Sample the eval_set_masked_contexts and ground_truth_citations using the random indices
    eval_set_masked_contexts = [eval_set_masked_contexts[i] for i in random_indices]
    ground_truth_citations = [ground_truth_citations[i] for i in random_indices]
# -------------------------- End of sampling eval set to 30000 documents ------------------------------------


# Create the retriever instance with the documents
retriever = CitationRetrieverBM25(docs)

# Retrieve top k documents for each masked context in the evaluation set
top_k_results = []
for m in tqdm(eval_set_masked_contexts, desc="Retrieving top k documents with BM25"):
    top_k_docs = retriever.retrieve_top_k(m, k=10)  # DIRECTLY RETRIEVE TOP 10 DOCUMENTS
    top_k_results.append(top_k_docs)

# Compare the retrieved citations with the ground truth citations
match_in_top_10_count = 0  # For Recall@10 calculation
match_ranks = []  # For MRR calculation (if no match, it will be 0)
exact_match_count = 0  # For exact match count
for i, top_k_docs in enumerate(top_k_results):
    retrieved_citations = [doc['citation'] for doc in top_k_docs]
    # Check for exact matches (when the ground truth citation is the first item in the retrieved list)
    if ground_truth_citations[i] in retrieved_citations[0]:  # Check only the first item for exact match
        exact_match_count += 1
    if ground_truth_citations[i] in retrieved_citations:
        match_in_top_10_count += 1
        match_ranks.append(retrieved_citations.index(ground_truth_citations[i]) + 1)  # +1 for 1-based index
    else:
        match_ranks.append(0)  # No match found

# Calculate the percentage of correct matches
recall_at_ten = (match_in_top_10_count / len(eval_set_masked_contexts)) * 100

# Calculate MRR score using the matched indices
mrr_score = sum((1 / idx if idx>0 else 0) for idx in match_ranks) / len(match_ranks)

# Calculate the percentage of exact matches
exact_match_percentage = (exact_match_count / len(eval_set_masked_contexts)) * 100

# Print the results
print(f"\nTotal number of correct top 10 matches: {match_in_top_10_count} out of {len(eval_set_masked_contexts)}")
print(f"Percentage of correct top 10 matches (Recall@10 score): {recall_at_ten:.2f}%")
print(f"Mean Reciprocal Rank (MRR) score: {mrr_score:.4f}")
print(f"Exact match count: {exact_match_count} out of {len(eval_set_masked_contexts)} --> {exact_match_percentage:.2f}%")
