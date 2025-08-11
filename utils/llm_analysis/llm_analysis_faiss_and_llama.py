# llm_analysis_faiss_and_LLM_fixed_v2.py
from typing import List, Dict
import pandas as pd
from tqdm import tqdm
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import torch
import random
import re
import os
import traceback
import logging
import warnings


"""
pip install fastbm25
pip install langchain
pip install langchain_community
pip install sentence-transformers
pip install accelerate==0.26.0
pip install faiss-cpu
pip install sentencepiece
pip install protobuf
(Also install PyTorch and transformers)
"""

warnings.filterwarnings("ignore", message=".*encoder_attention_mask.*is deprecated.*", category=FutureWarning)

# Transformers / HF hub
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    GenerationConfig,
)
from huggingface_hub import login as hf_login

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ====== USER FILE PATHS ======
eval_set_for_masked_contexts_file_path = "C:\\MY_FILES\\PycharmProjects\\CiteBART\\preprocessing\\global_datasets\\peerread_global\\context_dataset_eval.csv"
train_set_file_path = "C:\\MY_FILES\\PycharmProjects\\CiteBART\\preprocessing\\global_datasets\\peerread_global\\context_dataset_train.csv"
hf_token_file_path = "./HF_TOKEN.txt"

# ====== Read HuggingFace token ======
if not os.path.exists(hf_token_file_path):
    raise FileNotFoundError(f"HuggingFace token file not found at: {hf_token_file_path}")
with open(hf_token_file_path, 'r') as f:
    hf_token = f.read().strip()

# ====== Load documents ======
df_train = pd.read_csv(train_set_file_path)
temp_train_docs = df_train[["masked_token_target", "target_title", "target_abstract"]].to_dict(orient='records')

df_eval = pd.read_csv(eval_set_for_masked_contexts_file_path)
temp_eval_docs = df_eval[["masked_token_target", "target_title", "target_abstract"]].to_dict(orient='records')

# Merge and dedupe by citation
temp_docs = temp_train_docs + temp_eval_docs
docs = [
    {
        "citation": doc["masked_token_target"],
        "title": doc["target_title"],
        "abstract": doc["target_abstract"]
    }
    for doc in temp_docs
]
unique_docs = {doc['citation']: doc for doc in docs}.values()
docs = list(unique_docs)

print("\nTotal number of unique papers from the entire dataset:", len(docs), "\n")

# ====== Prepare eval contexts (sample small set for speed as before) ======
df_eval2 = pd.read_csv(eval_set_for_masked_contexts_file_path)
temp_eval_set_masked_contexts = df_eval2[["masked_cit_context", "citing_title", "citing_abstract"]].to_dict(orient='records')
ground_truth_citations = df_eval2["masked_token_target"].tolist()

eval_set_masked_contexts = [
    f"{d['masked_cit_context']} [SEP] {d['citing_title']} [SEP] {d['citing_abstract']}" for d in
    temp_eval_set_masked_contexts
]

random.seed(42)
if len(eval_set_masked_contexts) > 100:
    random_indices = random.sample(range(len(eval_set_masked_contexts)), 100)
    eval_set_masked_contexts = [eval_set_masked_contexts[i] for i in random_indices]
    ground_truth_citations = [ground_truth_citations[i] for i in random_indices]


class FAISSLLaMAPredictor:
    def __init__(
        self,
        all_docs: List[Dict],
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        llama_model: str = "meta-llama/Llama-2-7b-chat-hf",
        top_k: int = 10,
    ):
        self.top_k = top_k
        # ====== Embeddings & FAISS ======
        # (LangChain HuggingFaceEmbeddings is deprecated but functional; changing it is optional)
        logger.info("Initializing embeddings...")
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

        logger.info("Creating FAISS vector store from all documents...")
        documents = []
        for doc in all_docs:
            faiss_doc = Document(
                page_content=f"{doc['title']} {doc['abstract']}",
                metadata={
                    "citation": doc["citation"],
                    "title": doc["title"],
                    "abstract": doc["abstract"],
                },
            )
            documents.append(faiss_doc)
        self.vectorstore = FAISS.from_documents(documents, self.embeddings)
        logger.info("FAISS vector store created successfully!")

        # ====== HuggingFace authentication & model/tokenizer/pipeline ======
        # Use huggingface_hub.login() so we don't pass token into generation args
        logger.info("Logging into HuggingFace Hub...")
        try:
            hf_login(token=hf_token)
        except Exception as e:
            logger.warning("huggingface_hub.login() raised an exception (maybe already logged in): %s", e)

        logger.info("Loading tokenizer and model (this may take a while)...")
        # Load tokenizer and model explicitly so we control generation config (and avoid passing use_auth_token to generate)
        # trust_remote_code=True may be required for some community chat checkpoints — if you see an error, re-enable it.
        tokenizer = AutoTokenizer.from_pretrained(llama_model, use_fast=False)
        # Ensure tokenizer has a pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load model (device_map="auto" will place weights on available devices)
        # If memory is insufficient, this call will raise; you can adapt to cpu or 4-bit loading if desired.
        model = AutoModelForCausalLM.from_pretrained(
            llama_model,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
            # trust_remote_code=True,  # uncomment if the model needs it
        )

        # Set generation config on the model itself
        model.generation_config = GenerationConfig(
            max_new_tokens=50,
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
            repetition_penalty=1.1
        )

        logger.info("Creating text-generation pipeline...")
        hf_pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device_map="auto",
            max_new_tokens=50
            # No use_auth_token here (already logged in via hf_login)
        )

        # Wrap pipeline for LangChain
        from langchain_community.llms import HuggingFacePipeline
        self.llm = HuggingFacePipeline(pipeline=hf_pipe)

        # Prompt: strict XML output
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template=(
                "You are an automated citation selector. Given a masked citation context and a short list of candidate papers, "
                "select the single best candidate to replace the missing citation.\n\n"
                "Masked context:\n{question}\n\n"
                "Candidate papers (each line contains: citation - title - abstract):\n{context}\n\n"
                "IMPORTANT: Respond ONLY with EXACTLY ONE XML tag in this exact format (no extra text, no numbering):\n"
                "<citation>Author et al., YEAR</citation>\n\n"
                "Example valid response:\n<citation>Smith et al., 2020</citation>\n\n"
                "Now respond with the single citation selection."
            )
        )

        # LLMChain (still using LangChain's LLMChain wrapper)
        self.llm_chain = LLMChain(llm=self.llm, prompt=self.prompt_template)

    @staticmethod
    def _extract_tag(result: str) -> str:
        m = re.search(r"<citation>(.*?)</citation>", result, flags=re.IGNORECASE | re.DOTALL)
        if not m:
            return ""
        return " ".join(m.group(1).split())

    @staticmethod
    def _extract_author_year_simple(citation: str):
        """Return (author_part_lower, year)"""
        year_match = re.search(r'\b(19|20)\d{2}\b', citation)
        year = year_match.group() if year_match else ""
        if year:
            author_part = citation.split(year)[0].strip()
            author_part = re.sub(r'[,\s]+$', '', author_part)
        else:
            author_part = citation
        author_part = re.sub(r'\s+', ' ', author_part).strip()
        return author_part.lower(), year

    def _candidate_matches(self, candidate_list: List[str], chosen: str) -> bool:
        # direct match (case-insensitive)
        cand_lc = [c.lower().strip() for c in candidate_list]
        if chosen.lower().strip() in cand_lc:
            return True
        # otherwise match via author-year parsing
        chosen_author, chosen_year = self._extract_author_year_simple(chosen)
        for cand in candidate_list:
            a, y = self._extract_author_year_simple(cand)
            if (a == chosen_author) and (y == chosen_year) and (chosen_author != ""):
                return True
        return False

    def predict_single_citation(self, masked_context: str) -> str:
        try:
            # Retrieve top candidates from FAISS
            similar_docs = self.vectorstore.similarity_search(masked_context, k=self.top_k)
            candidates_text = ""
            citations_list = []
            for doc in similar_docs:
                citation = doc.metadata.get("citation", "").strip()
                citations_list.append(citation)
                candidates_text += f"[CANDIDATE] {citation}\nTitle: {doc.metadata.get('title', '')}\nAbstract: {doc.page_content}\n\n"

            # System + user message in LLaMA 2 chat format
            system_prompt = (
                "You are a citation selection system. Given a masked research paper context "
                "and a list of candidate citations, choose EXACTLY one citation that best fits the mask. "
                "Copy the citation EXACTLY as it appears in the candidate list. "
                "Respond ONLY in the following format:\n<citation>PASTE HERE</citation>"
            )
            user_prompt = f"Masked context:\n{masked_context}\n\nCandidate citations:\n{candidates_text}"

            llama_prompt = (
                f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_prompt} [/INST]"
            )

            # Run the LLaMA model
            result = self.llm(llama_prompt)

            # Extract only the response part (after [/INST])
            """if isinstance(result, list):
                full_output = result[0]['generated_text']
            elif hasattr(result, 'content'):
                full_output = result.content
            else:
                full_output = str(result)

            # Cut the initial part before [/INST] and extract only the actual response
            if "[/INST]" in full_output:
                actual_response = full_output.split("[/INST]")[-1].strip()
            else:
                actual_response = full_output

            # actual_response = result

            # Print the raw result for debugging
            ### print(f"*********************\n    LLM raw output: {actual_response} \n**********************\n")

            # Use actual_response for further processing
            output_text = actual_response"""

            if isinstance(result, list):
                output_text = result[0]['generated_text']
            elif hasattr(result, 'content'):
                output_text = result.content
            else:
                output_text = str(result)

            # Extract citation from tags - improved extraction
            matches = re.findall(r"<citation>(.*?)</citation>", output_text, re.DOTALL | re.IGNORECASE)
            citation_candidate = None
            for m in matches:
                if m.strip().lower() != "paste here":
                    citation_candidate = m.strip()
                    break

            if citation_candidate:
                ### print(f"Extracted citation: '{citation_candidate}'")

                # Method 1: Exact match (case-insensitive)
                for cand in citations_list:
                    if citation_candidate.lower().strip() == cand.lower().strip():
                        ### print(f"Found exact match: {cand}")
                        return cand

                # Method 2: Check if the extracted citation contains the candidate or vice versa
                for cand in citations_list:
                    if citation_candidate.lower().strip() in cand.lower().strip() or cand.lower().strip() in citation_candidate.lower().strip():
                        ### print(f"Found partial match: {cand}")
                        return cand

                # Method 3: Author-year matching as fallback
                cand_author, cand_year = self._extract_author_year_simple(citation_candidate)
                for cand in citations_list:
                    real_author, real_year = self._extract_author_year_simple(cand)
                    if (cand_author == real_author) and (cand_year == real_year) and (cand_author != ""):
                        ### print(f"Found author-year match: {cand}")
                        return cand

                ### print(f"No match found for '{citation_candidate}' in candidates:")
                ### for i, cand in enumerate(citations_list):
                    ### print(f"  {i + 1}. '{cand}'")

                ### logging.warning("LLM returned citation not matching candidates exactly. Falling back to FAISS top-1.")
                return citations_list[0]
            else:
                ### logging.warning(
                    ### "No <citation> tag found in LLM output or only 'PASTE HERE' found. Falling back to FAISS top-1.")
                return citations_list[0]


        except Exception as e:
            ### logging.error(f"Error during LLM run: {e}")
            traceback.print_exc()
            top_faiss_citation = self.vectorstore.similarity_search(masked_context, k=1)[0].metadata.get("citation", "")
            ### logging.warning(f"Falling back to top FAISS candidate due to exception: {top_faiss_citation}")
            return top_faiss_citation


# ====== Run evaluation ======
predictor = FAISSLLaMAPredictor(docs, top_k=10)

exact_match_count = 0
predictions = []

for i, masked_context in enumerate(tqdm(eval_set_masked_contexts, desc="Predicting citations")):
    predicted_citation = predictor.predict_single_citation(masked_context)
    predictions.append(predicted_citation)

    ### print(f"\nContext {i + 1}:")
    ### print(f"Predicted: {predicted_citation}")
    ### print(f"Ground Truth: {ground_truth_citations[i]}")

    # match via author/year
    pred_author, pred_year = predictor._extract_author_year_simple(predicted_citation)
    truth_author, truth_year = predictor._extract_author_year_simple(ground_truth_citations[i])

    if (pred_author == truth_author) and (pred_year == truth_year):
        exact_match_count += 1
        ### print("✓ MATCH!")
    ### else:
        ### print("✗ No match")
        ### print(f"  Predicted parsed: '{pred_author}', '{pred_year}'")
        ### print(f"  Ground truth parsed: '{truth_author}', '{truth_year}'")

print(f"\nExact matches: {exact_match_count} / {len(eval_set_masked_contexts)}")
