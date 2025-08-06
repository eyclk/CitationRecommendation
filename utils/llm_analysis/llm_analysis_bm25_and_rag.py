from fastbm25 import fastbm25
from typing import List, Dict
import pandas as pd
from tqdm import tqdm
from langchain_community.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
import torch


eval_set_for_masked_contexts_file_path = "C:\MY_FILES\PycharmProjects\CiteBART\preprocessing\global_datasets\\peerread_global\context_dataset_eval.csv"

train_set_file_path = "C:\MY_FILES\PycharmProjects\CiteBART\preprocessing\global_datasets\\peerread_global\context_dataset_train.csv"

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

# eval_set_masked_contexts = eval_set_masked_contexts[:5]  # -----------------------------------------------------------
# ground_truth_citations = ground_truth_citations[:5]


# Create the retriever instance with the documents
retriever = CitationRetrieverBM25(docs)

# Retrieve top k documents for each masked context in the evaluation set
top_k_results = []
for m in tqdm(eval_set_masked_contexts, desc="Retrieving top k documents with BM25"):
    top_k_docs = retriever.retrieve_top_k(m, k=100)  # Prefetch 100 documents for each masked context !!!!!!!!!!!!!!!!!!!
    top_k_results.append(top_k_docs)

# Merge the top k results into a single string for each document
top_k_results_merged_strings = [
    [f"Author-Date Citation: {doc['citation']} [SEP] Title: {doc['title']} [SEP] Abstract: {doc['abstract']}" for doc in top_k_docs]
    for top_k_docs in top_k_results
]


# **********************************************************************************************************************


# Define the LangChain RAG model for citation prediction
class LangChainRAGCitationPredictor:
    def __init__(self, model_path: str = None, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

        # Initialize LLM with better parameters
        if model_path:
            self.llm = LlamaCpp(
                model_path=model_path,
                temperature=0.3,
                max_tokens=200,  # Increased for multiple citations
                n_ctx=4096,
                verbose=False
            )
        else:
            from langchain_community.llms import HuggingFacePipeline
            from transformers import pipeline

            pipe = pipeline(
                "text-generation",
                model="meta-llama/Llama-2-7b-chat-hf",
                token=hf_token,
                torch_dtype=torch.float16,
                device_map="auto",
                max_new_tokens=120,  # Increased for multiple citations
                temperature=0.3,
                do_sample=True,
                pad_token_id=50256,
                repetition_penalty=1.1
            )
            self.llm = HuggingFacePipeline(pipeline=pipe)

        # Updated prompt for multiple citations
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""Given this context with a missing citation: {question}

        Here are relevant papers:
        {context}

        Select the 10 most relevant citations to replace <mask>. Respond with ONLY the citation format (Author et al., Year):

        1. """
        )

        # Create LLM chain
        self.llm_chain = LLMChain(llm=self.llm, prompt=self.prompt_template)

    def predict_citations(self, masked_context: str, retrieved_docs: List[str], num_predictions: int = 10) -> List[str]:
        """Use FAISS vector store for better retrieval from 100 docs"""

        # Create documents for FAISS from the 100 retrieved docs
        documents = []
        for doc in retrieved_docs:
            citation = doc.split("Author-Date Citation:")[1].split("[SEP]")[0].strip()
            title = doc.split("Title:")[1].split("[SEP]")[0].strip()
            abstract = doc.split("Abstract:")[1].strip()

            # Create document with metadata - use abstract as main content for similarity
            faiss_doc = Document(
                page_content=abstract,
                metadata={
                    "citation": citation,
                    "title": title,
                    "full_doc": doc
                }
            )
            documents.append(faiss_doc)

        # Create FAISS vector store from these 100 docs
        vectorstore = FAISS.from_documents(documents, self.embeddings)

        # Retrieve top 10 most similar documents based on semantic similarity
        similar_docs = vectorstore.similarity_search(masked_context, k=10)

        # Extract top 10 docs and create concise context
        context_parts = []
        for i, doc in enumerate(similar_docs):
            citation = doc.metadata['citation']
            title = doc.metadata['title']
            abstract = doc.page_content[:200] + "..."

            # Make it clear this is a paper option
            context_parts.append(f"Paper {i + 1}: Citation: {citation} | Title: {title} | Abstract: {abstract}")

        context = "\n".join(context_parts)

        # Generate prediction with LLM
        try:
            result = self.llm_chain.run(context=context, question=masked_context)

            """print("-------- RAG Prediction Result --------")
            print(result)
            print("---------------------------------------")"""

            # Parse predictions with better extraction
            predictions = []
            lines = result.strip().split('\n')

            for line in lines:
                line = line.strip()
                if line and any(line.startswith(f"{i}.") for i in range(1, 11)):
                    # Extract everything after the number and period
                    citation_part = line.split(".", 1)[1].strip() if "." in line else line

                    # Clean up - remove everything after the first |, [, or other delimiter
                    if "|" in citation_part:
                        citation_part = citation_part.split("|")[0].strip()
                    if "[" in citation_part:
                        citation_part = citation_part.split("[")[0].strip()
                    if "Title:" in citation_part:
                        citation_part = citation_part.split("Title:")[0].strip()

                    # Only keep if it looks like a proper citation (has "et al." or comma and year)
                    if citation_part and (("et al." in citation_part) or (
                            ", " in citation_part and any(c.isdigit() for c in citation_part))):
                        if citation_part not in predictions:
                            predictions.append(citation_part)

            # If we don't have enough valid predictions, fall back to FAISS-ranked citations
            while len(predictions) < num_predictions:
                faiss_citations = [doc.metadata['citation'] for doc in similar_docs]
                for citation in faiss_citations:
                    if citation not in predictions and len(predictions) < num_predictions:
                        predictions.append(citation)

            return predictions[:num_predictions]

        except Exception as e:
            print(f"Error in prediction: {e}")
            # Complete fallback to top similarity-ranked citations
            return [doc.metadata['citation'] for doc in similar_docs[:num_predictions]]


# Read HuggingFace access token from the "HF_TOKEN.txt" file
hf_token_file_path = ".\HF_TOKEN.txt"
with open(hf_token_file_path, 'r') as file:
    hf_token = file.read().strip()


# Initialize the LangChain RAG citation predictor
rag_predictor = LangChainRAGCitationPredictor()

"""
# Predict 10 citations for the first context
rag_predictions = rag_predictor.predict_citations(
    eval_set_masked_contexts[1],          ###TEMPORARY TESTING  -------- After testing, change it to loop over all contexts
    top_k_results_merged_strings[1],
    num_predictions=10
)

print(f"Generated {len(rag_predictions)} RAG predictions")

# Print the predictions line by line
for i, prediction in enumerate(rag_predictions):
    print(f"Prediction {i + 1}: {prediction}")

print(f"\nGround Truth Citation: {ground_truth_citations[1]}")
"""

# Loop over all masked contexts in the evaluation set (eval_set_masked_contexts) and their corresponding top k results (top_k_results_merged_strings)
# Pass them to the RAG predictor to get the predicted citations and compare with ground truth citations
match_in_top_10_count = 0  # For Recall@10 calculation
match_ranks = []  # For MRR calculation (if no match, it will be 0)
exact_match_count = 0  # For exact match count
for i, masked_context in enumerate(tqdm(eval_set_masked_contexts, desc="Predicting citations with RAG")):
    # Get the top k results for this masked context
    top_k_docs = top_k_results_merged_strings[i]

    # Predict citations using the RAG model
    rag_predictions = rag_predictor.predict_citations(
        masked_context,
        top_k_docs,
        num_predictions=10
    )

    # Check if the ground truth citation is in the predicted citations
    if ground_truth_citations[i] in rag_predictions:
        match_in_top_10_count += 1

    # Calculate rank for MRR
    try:
        rank = rag_predictions.index(ground_truth_citations[i]) + 1  # +1 for 1-based index
        match_ranks.append(rank)
    except ValueError:
        match_ranks.append(0)  # No match found

    # Check for exact match (when the ground truth citation is the first item in the predicted list)
    if rag_predictions and ground_truth_citations[i] == rag_predictions[0]:
        exact_match_count += 1

# Calculate the percentage of correct matches
recall_at_ten = (match_in_top_10_count / len(eval_set_masked_contexts)) * 100
# Calculate MRR score using the matched indices
mrr_score = sum((1 / idx if idx > 0 else 0) for idx in match_ranks) / len(match_ranks)
# Calculate the percentage of exact matches
exact_match_percentage = (exact_match_count / len(eval_set_masked_contexts)) * 100

# Print the results
print(f"\nTotal number of correct top 10 matches: {match_in_top_10_count} out of {len(eval_set_masked_contexts)}")
print(f"Percentage of correct top 10 matches (Recall@10 score): {recall_at_ten:.2f}%")
print(f"Mean Reciprocal Rank (MRR) score: {mrr_score:.4f}")
print(f"Exact match count: {exact_match_count} out of {len(eval_set_masked_contexts)} --> {exact_match_percentage:.2f}%")

