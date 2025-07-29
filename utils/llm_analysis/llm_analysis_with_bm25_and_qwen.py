from fastbm25 import fastbm25
from typing import List, Dict
# import nltk
# nltk.download('punkt')
# from nltk.tokenize import word_tokenize
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


all_documents_file_path = "C:\MY_FILES\PycharmProjects\CiteBART\preprocessing\global_datasets\peerread_global\context_dataset.csv"

eval_set_for_masked_contexts_file_path = "C:\MY_FILES\PycharmProjects\CiteBART\preprocessing\global_datasets\peerread_global\context_dataset_eval.csv"

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
        return [self.documents[i[1]] for i in top_indices]


"""     ### EXAMPLE USAGE ###
docs = [{"citation": "Vinyals et al., 2014", "title": "Grammar as a Foreign Language", "abstract": "Syntactic parsing is a fundamental problem in computational linguistics and Natural Language Processing. Traditional approaches to parsing are highly complex and problem specific. Recently, Sutskever et al. (2014) presented a domain-independent method for learning to map input sequences to output sequences that achieved strong results on a large scale machine translation problem. In this work, we show that precisely the same sequence-to-sequence method achieves results that are close to state-of-the-art on syntactic constituency parsing, whilst making almost no assumptions about the structure of the problem."},
        {"citation": "Sutskever et al., 2014", "title": "Sequence to Sequence Learning with Neural Networks", "abstract": "Deep Neural Networks (DNNs) are powerful models that have achieved excellent performance on difficult learning tasks. Although DNNs work well whenever large labeled training sets are available, they cannot be used to map sequences to sequences. In this paper, we present a general end-to-end approach to sequence learning that makes minimal assumptions on the sequence structure. Our method uses a multilayered Long Short-Term Memory (LSTM) to map the input sequence to a vector of a fixed dimensionality, and then another deep LSTM to decode the target sequence from the vector. Our main result is that on an English to French translation task from the WMT-14 dataset, the translations produced by the LSTM achieve a BLEU score of 34.7 on the entire test set, where the LSTM's BLEU score was penalized on out-of-vocabulary words. Additionally, the LSTM did not have difficulty on long sentences. For"}]
masked_citing_context = "nd-crafted features, lexicons, and grammars.Meanwhile, recurrent neural networks  have made swift inroads intomany structured prediction tasks in NLP,including machine translation andsyntactic parsing  <mask>  .Because RNNs make very few domain-specific assumptions,they have the potential to succeed at a wide variety of taskswith minimal feature engineering.wever, this flexibility also puts RNNs at a disadv"
retriever = CitationRetrieverBM25(docs)
top_k_docs = retriever.retrieve_top_k(masked_citing_context, k=1)
print(top_k_docs)
"""

# Open all documents csv file and read the documents
df = pd.read_csv(all_documents_file_path)
# Convert to list of dictionaries. From df, the column names are "masked_token_target", "target_title", "target_abstract"
temp_docs = df[["masked_token_target", "target_title", "target_abstract"]].to_dict(orient='records')

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

"""example_masked_context = "software,requiring hand-crafted features, lexicons, and grammars.Meanwhile, recurrent neural networks  have made swift inroads intomany structured prediction tasks in NLP,including machine translation  <mask>  andsyntactic parsing .Because RNNs make very few domain-specific assumptions,they have the potential to succeed at a wide variety of taskswith minimal feature engineering.wever, this flexibility also "

retriever = CitationRetrieverBM25(docs)
top_k_docs = retriever.retrieve_top_k(example_masked_context, k=100)  # Try with 100 and 300

# Print the top k documents line by line
for i, doc in enumerate(top_k_docs):
    print(f"Top {i+1} Document:")
    print(f"Citation: {doc['citation']}")
    print(f"Title: {doc['title']}")
    print(f"Abstract: {doc['abstract']}\n\n")"""

df_eval = pd.read_csv(eval_set_for_masked_contexts_file_path)

temp_eval_set_masked_contexts = df_eval[["masked_cit_context", "citing_title", "citing_abstract"]].to_dict(orient='records')

# These are the ground truth citations for the evaluation set. After LLM model prediction, we will compare the predicted citations with these ground truth citations.
ground_truth_citations = df_eval["masked_token_target"].tolist()

eval_set_masked_contexts = [
    f"{d['masked_cit_context']} [SEP] {d['citing_title']} [SEP] {d['citing_abstract']}" for d in temp_eval_set_masked_contexts
]


# Create the retriever instance with the documents
retriever = CitationRetrieverBM25(docs)

# Retrieve top k documents for each masked context in the evaluation set
top_k_results = []
for masked_context in eval_set_masked_contexts:
    top_k_docs = retriever.retrieve_top_k(masked_context, k=100)  # Retrieve top 100 or 300 documents
    top_k_results.append(top_k_docs)
"""
# Print the top k results line by line for the first masked context as an example
for i, doc in enumerate(top_k_results[0]):
    print(f"Top {i+1} Document:")
    print(f"Citation: {doc['citation']}")
    print(f"Title: {doc['title']}")
    print(f"Abstract: {doc['abstract']}\n\n\n")

# Print the first masked context as an example alongside its ground truth citation
print("Masked Context:", eval_set_masked_contexts[0], "\n")
print("Ground Truth Citation:", ground_truth_citations[0])
"""

top_k_results_merged_strings = [
    [f"Citation: {doc['citation']} [SEP] Title: {doc['title']} [SEP] Abstract: {doc['abstract']}" for doc in top_k_docs]
    for top_k_docs in top_k_results
]


# ************************************************** QWEN ZERO-SHOT ANALYSIS **************************************

def format_prompt(context: str, candidate: str) -> str:
    return f"You are a local citation recommender. Based on the relevance between the query “{context}” and the document “{candidate}”, assign a single digit score between 0 and 100. Please provide only the score as the output."



# Load Qwen model (e.g., Qwen-1.5-Chat)
model_name = "Qwen/Qwen-1.5-7B-Chat"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", trust_remote_code=True)


def score_candidate(prompt: str) -> float:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=128)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract score (e.g., look for "Score: 4" or "4 out of 5")
    """import re
    match = re.search(r"\b([1-100])\b", response)
    if match:
        return int(match.group(1))
    else:
        return 0  # fallback if no number found"""

    # Extract score. It should be a number between 0 and 100. Convert it to float if it is between 0 and 100. Otherwise, return 0.
    try:
        score = float(response.strip())
        if 0 <= score <= 100:
            return score
        else:
            return 0.0  # fallback if score is out of range
    except ValueError:
        return 0.0



#############   BURALARI DÜZELT   !!!!!!!!!!!


# Loop through candidates
def select_best_citation(context: str, candidates: List[str]) -> str:
    best_score = -1
    best_candidate = None
    for cand in candidates:
        prompt = format_prompt(context, cand)
        score = score_candidate(prompt)
        if score > best_score:
            best_score = score
            best_candidate = cand
    return best_candidate


### EXAMPLE USAGE ###
best = select_best_citation(temp_masked_context, top_k_citations_for_masked_context)
print("Best Matching Citation Title:", best)


