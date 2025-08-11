from fastbm25 import fastbm25
from typing import List, Dict
# import nltk
# nltk.download('punkt')
# from nltk.tokenize import word_tokenize
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm
import re


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

print("\nTotal number of unique papers from the entire dataset:", len(docs), "\n")

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
for m in tqdm(eval_set_masked_contexts, desc="Retrieving top k documents with BM25"):
    top_k_docs = retriever.retrieve_top_k(m, k=100)  # Retrieve top 100 or 300 documents
    top_k_results.append(top_k_docs)

# Merge the top k results into a single string for each document
top_k_results_merged_strings = [
    [f"{doc['citation']} [SEP] Title: {doc['title']} [SEP] Abstract: {doc['abstract']}" for doc in top_k_docs]
    for top_k_docs in top_k_results
]


# ************************************************** QWEN ZERO-SHOT ANALYSIS **************************************


def format_prompt(context: str, candidate: str) -> str:
    return f"You are a local citation recommender. Based on the relevance between the query “{context}” and the document “{candidate}”, assign a numerical score between 0 and 100. Please provide only the score as the output."


"""
######  ALTERNATIVE PROMPT FORMAT WITH SINGLE DIGIT WORDING ######
def format_prompt(context: str, candidate: str) -> str:
    return f"You are a local citation recommender. Based on the relevance between the query “{context}” and the document “{candidate}”, assign a single digit score between 0 and 100. Please provide only the score as the output."
"""


# Load Qwen model (e.g., Qwen1.5-Chat)
model_name = "Qwen/Qwen1.5-1.8B-Chat"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to("cuda")  ## device_map="auto"  , device_map={"": "cuda"}


def batch_score_candidates(context: str, candidates: List[str], batch_size: int = 8) -> List[float]:
    scores = []

    # for i in tqdm(range(0, len(candidates), batch_size), desc="Scoring candidates"):
    for i in range(0, len(candidates), batch_size):
        batch = candidates[i:i + batch_size]
        prompts = [format_prompt(context, cand) for cand in batch]

        # Tokenize as chat format
        messages = [{"role": "user", "content": prompt} for prompt in prompts]
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(model.device)

        with torch.no_grad():
            torch.manual_seed(42)  # For reproducibility across runs
            outputs = model.generate(**inputs, max_new_tokens=32, do_sample=False,  top_p=None)  # max_new_tokens can be 8 or 32. However, 32 is better.

        batch_responses = tokenizer.batch_decode(
            [output[input_ids.shape[-1]:] for output, input_ids in zip(outputs, inputs['input_ids'])],
            skip_special_tokens=True
        )

        print("**  ", batch_responses)   ##########

        for response in batch_responses:
            try:
                score = float(response.strip())
                if 0 <= score <= 100:
                    scores.append(score)
                    continue
            except ValueError:
                pass

            # Fallback: try to extract a float
            extracted_score = re.search(r"\d+(\.\d+)?", response.strip())
            if extracted_score:
                score = float(extracted_score.group(0))
                if 0 <= score <= 100:
                    scores.append(score)
                    continue

            scores.append(0.0)  # fallback

    print(f"\nReturned {len(scores)} scores for batch size {batch_size} for a given context and its 100 candidates.\n")  # Debugging output

    return scores


def select_top_10_citations(context: str, candidates: List[str], batch_size: int = 8):
    scores = batch_score_candidates(context, candidates, batch_size=batch_size)

    # Select top 10 candidates based on scores
    top_10_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:10]
    top_10_candidates = [candidates[i] for i in top_10_indices]
    top_10_scores = [scores[i] for i in top_10_indices]

    print("\n---- Top 10 candidates and their scores: ")
    for i, (candidate, score) in enumerate(zip(top_10_candidates, top_10_scores), start=1):
        print(f"{i}. (Score: {score}) --> {candidate} ")
    print("---- End of Top 10 candidates and their scores.\n")

    return top_10_candidates, top_10_scores


### EXAMPLE USAGE ###
"""temp_masked_context = eval_set_masked_contexts[0]  # Example masked context
top_k_citations_for_masked_context = top_k_results_merged_strings[0]  # Top k citations for the example masked context
best = select_best_citation(temp_masked_context, top_k_citations_for_masked_context, batch_size=4)

print("Best Matching Citation Title:", best)
print("\nGround Truth Citation Title:", ground_truth_citations[0])"""


eval_set_masked_contexts = eval_set_masked_contexts[:10]  # LIMIT to first 100 for testing ......... TEMP


correct_top_10_match_count = 0
matched_top_10_indices = []
exact_match_count = 0
for e in tqdm(range(len(eval_set_masked_contexts)), desc="Processing the entire evaluation set"):
    temp_masked_context = eval_set_masked_contexts[e]  # Example masked context
    top_k_citations_for_masked_context = top_k_results_merged_strings[e]  # Top k citations for the example masked context

    # Print temp_masked_context and ground truth for debugging
    print(f"\n\nProcessing masked context {e+1}/{len(eval_set_masked_contexts)}:")
    print(f"Masked Context: {temp_masked_context}")
    print(f"Ground Truth Citation: {ground_truth_citations[e]}")

    temp_top_10, _ = select_top_10_citations(temp_masked_context, top_k_citations_for_masked_context, batch_size=1)  ## batch_size=1 IS NECESSARY !!!!

    print("\n\n")

    # Extract the citation from the top 10 results by splitting on [SEP] and taking the first part
    temp_top_10_citations = [doc.split("[SEP]")[0].strip() for doc in temp_top_10]

    # Check if the ground truth citation matches any of the top 10 citations
    if ground_truth_citations[e] == temp_top_10_citations[0]:
        exact_match_count += 1

    match_found = False
    # Check if the top 10 citation matches the ground truth citation
    for t in range(len(temp_top_10_citations)):
        if temp_top_10_citations[t] == ground_truth_citations[e]:
            correct_top_10_match_count += 1
            matched_top_10_indices.append(t + 1)
            match_found = True
            break

    if not match_found:
        matched_top_10_indices.append(0)

# Calculate the percentage of correct matches
correct_percentage = (correct_top_10_match_count / len(eval_set_masked_contexts)) * 100

# Calculate MRR score using the matched indices
mrr_score = sum((1 / idx if idx>0 else 0) for idx in matched_top_10_indices) / len(matched_top_10_indices)

# Calculate the percentage of exact matches
exact_match_percentage = (exact_match_count / len(eval_set_masked_contexts)) * 100

# Print the results
print(f"\nTotal number of correct top 10 matches: {correct_top_10_match_count} out of {len(eval_set_masked_contexts)}")
print(f"Percentage of correct top 10 matches (Recall@10 score): {correct_percentage:.2f}%")
print(f"Mean Reciprocal Rank (MRR) score: {mrr_score:.4f}")
print(f"Exact match count: {exact_match_count} out of {len(eval_set_masked_contexts)} --> {exact_match_percentage:.2f}%")


################ NOTE TO SELF: REMOVE ANY UNNECESSARY PRINT STATEMENTS ############
