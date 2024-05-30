from transformers import RobertaForMaskedLM, RobertaTokenizer, pipeline
from datasets import Dataset
import pandas as pd
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--trained_model_path", type=str, help="Path of the local trained model")
parser.add_argument("--dataset_path", type=str, default="", help="Path to the folder of the dataset")
parser.add_argument("--eval_path", type=str, help="Path to the evaluation set of the dataset")
parser.add_argument("--vocab_additions_path", type=str, help="Path to the additional vocab file of the dataset")
parser.add_argument("--max_token_limit", type=int, default=400, help="Max amount allowed for tokens used evaluation")
parser.add_argument("--output_file", type=str, default="./outputs/eval_results.txt", help="Path to file that will "
                                                                                          "contain outputs and results")


def tokenizer_function(tknizer, inp_data, col_name):
    return tknizer(inp_data[col_name], truncation=True, padding='max_length', max_length=max_token_limit)


def read_eval_dataset(tknizer):
    cit_df = pd.read_csv(eval_set_path)
    input_texts = []
    label_texts = []
    masked_targets = []

    for _, i in cit_df.iterrows():
        input_texts.append(i['masked_cit_context'])
        label_texts.append(i['citation_context'])
        masked_targets.append(i['masked_token_target'])

    df_text_list = pd.DataFrame(input_texts, columns=['input_ids'])
    data_input_ids = Dataset.from_pandas(df_text_list)
    tokenized_input_ids = data_input_ids.map(
        lambda batch: tokenizer_function(tknizer, batch, 'input_ids'), batched=True)

    df_label_list = pd.DataFrame(label_texts, columns=['labels'])
    data_labels = Dataset.from_pandas(df_label_list)
    tokenized_labels = data_labels.map(
        lambda batch: tokenizer_function(tknizer, batch, 'labels'), batched=True)

    tokenized_data = tokenized_input_ids.add_column('labels', tokenized_labels['input_ids'])

    raw_and_tokenized_data = tokenized_data.add_column('masked_cit_context', input_texts)
    raw_and_tokenized_data = raw_and_tokenized_data.add_column('citation_context', label_texts)
    raw_and_tokenized_data = raw_and_tokenized_data.add_column('masked_token_target', masked_targets)

    return raw_and_tokenized_data


def shorten_masked_context_for_limit_if_necessary(masked_text):
    tokenized_text = tokenizer.tokenize(masked_text)
    if len(tokenized_text) > max_token_limit:  # Shorten texts with more than the max limit.
        exceeding_char_count = 5  # Always start with 5 extra characters just in case.
        for i in range(max_token_limit-1, len(tokenized_text)):
            exceeding_char_count += len(tokenized_text[i])
        shortened_text = masked_text[:-exceeding_char_count]
        return shortened_text
    return masked_text


# This is the same thing as recall@10. Recall@10 can only found values 0/1 or 1/1. So, it is either hit or miss.
def calc_hits_at_k_score(k=10):
    hit_count = 0
    pred_comparison_count = 0
    for j in range(len(top_10_cit_preds)):
        pred_comparison_count += 1
        preds = top_10_cit_preds[j]
        for pr in preds:
            if pr == masked_token_targets[j]:
                hit_count += 1
                break

    hit_at_k_metric = hit_count / pred_comparison_count
    print(f"\n=======>>> Hits@{k} score (between 0 and 1) = ", hit_at_k_metric, "\n")
    f_out.write(f"\n=======>>> Hits@{k} score (between 0 and 1) = {hit_at_k_metric}\n")


def calc_exact_match_acc_score():
    exact_match_count = 0
    pred_comparison_count = 0
    for j in range(len(top_10_cit_preds)):
        pred_comparison_count += 1

        if top_10_cit_preds[j][0] == masked_token_targets[j]:
            exact_match_count += 1

    exact_match_metric = exact_match_count / pred_comparison_count
    print("\n=======>>> Exact match/accuracy score (between 0 and 1) = ", exact_match_metric, "\n")
    f_out.write(f"\n=======>>> Exact match/accuracy score (between 0 and 1) = {exact_match_metric}\n")


def calc_mrr_score():
    reciprocal_rank_list = []

    for j in range(len(top_10_cit_preds)):
        reciprocal_rank_list.append(0)  # Start all recip ranks as 0. If match is found, then it is replaced.
        preds = top_10_cit_preds[j]

        for p_idx in range(len(preds)):
            if preds[p_idx] == masked_token_targets[j]:
                temp_reciprocal_rank = 1 / (p_idx + 1)
                reciprocal_rank_list[-1] = temp_reciprocal_rank
                break

    mean_reciprocal_rank = np.mean(reciprocal_rank_list)
    print("\n=======>>> MRR score = ", mean_reciprocal_rank, "\n")
    f_out.write(f"\n=======>>> MRR score = {mean_reciprocal_rank}\n")


def complete_calc_hits_at_k_score(k=10):
    hit_count = 0
    pred_comparison_count = 0
    for j in range(len(all_preds_generic)):
        pred_comparison_count += 1
        preds = all_preds_generic[j]
        for pr in preds:
            if pr == masked_token_targets[j]:
                hit_count += 1
                break

    hit_at_k_metric = hit_count / pred_comparison_count
    print(f"\n=======>>> Hits@{k} score (between 0 and 1) = ", hit_at_k_metric, "\n")
    f_out.write(f"\n=======>>> Hits@{k} score (between 0 and 1) = {hit_at_k_metric}\n")


def complete_calc_exact_match_acc_score():
    exact_match_count = 0
    pred_comparison_count = 0
    for j in range(len(all_preds_generic)):
        pred_comparison_count += 1

        if all_preds_generic[j][0] == masked_token_targets[j]:
            exact_match_count += 1

    exact_match_metric = exact_match_count / pred_comparison_count
    print("\n=======>>> Exact match/accuracy score (between 0 and 1) = ", exact_match_metric, "\n")
    f_out.write(f"\n=======>>> Exact match/accuracy score (between 0 and 1) = {exact_match_metric}\n")


def complete_calc_mrr_score():
    reciprocal_rank_list = []

    for j in range(len(all_preds_generic)):
        reciprocal_rank_list.append(0)  # Start all recip ranks as 0. If match is found, then it is replaced.
        preds = all_preds_generic[j]

        for p_idx in range(len(preds)):
            if preds[p_idx] == masked_token_targets[j]:
                temp_reciprocal_rank = 1 / (p_idx + 1)
                reciprocal_rank_list[-1] = temp_reciprocal_rank
                break

    mean_reciprocal_rank = np.mean(reciprocal_rank_list)
    print("\n=======>>> MRR score = ", mean_reciprocal_rank, "\n")
    f_out.write(f"\n=======>>> MRR score = {mean_reciprocal_rank}\n")


if __name__ == '__main__':
    args = parser.parse_args()

    local_model_path = args.trained_model_path
    max_token_limit = args.max_token_limit

    dataset_folder = args.dataset_path
    if dataset_folder == "":
        additional_vocab_path = args.vocab_additions_path
        eval_set_path = args.eval_path
    else:
        additional_vocab_path = dataset_folder + "/additions_to_vocab.csv"
        eval_set_path = dataset_folder + "/context_dataset_eval.csv"

    f_out = open(args.output_file, "w")

    tokenizer = RobertaTokenizer.from_pretrained(local_model_path, truncation=True, padding='max_length',
                                                 max_length=max_token_limit)
    model = RobertaForMaskedLM.from_pretrained(local_model_path)

    print("*** Added the new citations tokens to the tokenizer. Example for acl-200:\n",
          tokenizer.tokenize('Our paper is referencing the paper of Nenkova and Passonneau, 2004'), "\n\n")
    print("*** Another example for peerread:\n",
          tokenizer.tokenize('Our paper is referencing the paper of Gribkoff et al., 2014'), "\n\n")

    print("*** Another example for refseer:\n",
          tokenizer.tokenize('Our paper is referencing the paper of Lecoutre and Boussemart, 2003'), "\n\n")
    print("*** Another example for refseer:\n",
          tokenizer.tokenize('Our paper is referencing the paper of Dawson and Jahanian, 1995'), "\n\n")
    print("*** Another example for arxiv:\n",
          tokenizer.tokenize('Our paper is referencing the paper of Fishman et al., 2009'), "\n\n")
    print("*** Another example for arxiv:\n",
          tokenizer.tokenize('Our paper is referencing the paper of Vodolazov and Peeters, 2007'), "\n\n")

    eval_dataset = read_eval_dataset(tokenizer)

    mask_filler = pipeline(
        "fill-mask", model=model, tokenizer=tokenizer, top_k=100, device=0
    )
    cit_df_for_test = eval_dataset.to_pandas()

    input_texts_for_test = []
    masked_token_targets = []
    for _, cit in cit_df_for_test.iterrows():
        temp_masked_text = cit["masked_cit_context"]

        temp_masked_text = shorten_masked_context_for_limit_if_necessary(temp_masked_text)
        if temp_masked_text.find("<mask>") == -1:
            continue
        input_texts_for_test.append(temp_masked_text)

        masked_token_targets.append(cit['masked_token_target'])

    print("\nPROGRESS: Pipeline mask_filler is STARTING for top 100 predictions.\n")

    all_preds = mask_filler(input_texts_for_test)

    print("\nPROGRESS: Pipeline mask_filler is COMPLETED for top 100 predictions.\n")

    print("\nPROGRESS: Selection of only top 10 citations out of top 100 predictions is STARTING.\n")

    additional_vocab_df = pd.read_csv(additional_vocab_path)
    additional_vocab_list = list(additional_vocab_df["additions_to_vocab"])

    top_10_cit_preds = []
    for t in range(len(all_preds)):
        temp_top_10_cit = []
        temp_predictions = all_preds[t]

        for r in temp_predictions:
            if isinstance(r, list):
                for pred_in in r:
                    if pred_in['token_str'] in additional_vocab_list:
                        temp_top_10_cit.append(pred_in['token_str'])
                        continue
            elif r['token_str'] in additional_vocab_list:
                temp_top_10_cit.append(r['token_str'])

            if len(temp_top_10_cit) == 10:
                break

        top_10_cit_preds.append(temp_top_10_cit)

    print("\nPROGRESS: Selection of only top 10 citations out of top 100 predictions is COMPLETED.\n")

    print("~" * 40)
    print("\n*** Calculating Hits@10 score")
    calc_hits_at_k_score(k=10)

    print("~" * 40)
    print("\n*** Calculating Exact Match/Accuracy score")
    calc_exact_match_acc_score()

    print("~" * 40)
    print("\n*** Calculating MRR score")
    calc_mrr_score()

    """print("~" * 40)
    print("\n*** Calculating Recall@10 score")
    calc_recall_at_k_score(k=10)"""

    print("\n========\n\nPROGRESS: STARTING calculation of metrics for predictions with complete vocabulary.\n")

    all_preds_generic = []
    for t in range(len(all_preds)):
        temp_preds_list = []
        temp_predictions = all_preds[t]

        for r in temp_predictions:
            if isinstance(r, list):
                for pred_in in r:
                    temp_preds_list.append(pred_in['token_str'])
                    if len(temp_preds_list) == 10:
                        break
            else:
                temp_preds_list.append(r['token_str'])

            if len(temp_preds_list) == 10:
                break

        all_preds_generic.append(temp_preds_list)

    print("\n\n=====================> Results of the evaluation with all predictions instead of only citations:\n")

    print("~" * 40)
    print("\n*** Calculating Hits@10 (generic) score")
    complete_calc_hits_at_k_score(k=10)

    print("~" * 40)
    print("\n*** Calculating Exact Match/Accuracy (generic) score")
    complete_calc_exact_match_acc_score()

    print("~" * 40)
    print("\n*** Calculating MRR (generic) score")
    complete_calc_mrr_score()

    f_out.close()
