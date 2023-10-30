from transformers import RobertaForMaskedLM, RobertaTokenizer, pipeline
from datasets import Dataset
import pandas as pd
import numpy as np


local_model_path = "./models/cit_pred_v2_arxiv300k"
eval_set_path = "./cit_data/arxiv300k/context_dataset_eval.csv"
additional_vocab_path = "./cit_data/arxiv300k/additions_to_vocab.csv"
max_token_limit = 400


def tokenizer_function(tknizer, inp_data, col_name):
    return tknizer(inp_data[col_name], truncation=True, padding='max_length', max_length=max_token_limit)


def read_eval_dataset(tknizer):
    cit_df = pd.read_csv(eval_set_path)
    input_texts = []
    label_texts = []
    masked_token_targets = []

    for _, i in cit_df.iterrows():
        input_texts.append(i['masked_cit_context'])
        label_texts.append(i['citation_context'])
        masked_token_targets.append(i['masked_token_target'])

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
    raw_and_tokenized_data = raw_and_tokenized_data.add_column('masked_token_target', masked_token_targets)

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
def calc_hits_at_k_score(val_dataset, k=10):
    mask_filler = pipeline(
        "fill-mask", model=model, tokenizer=tokenizer, top_k=k, device=0
    )
    cit_df_for_test = val_dataset.to_pandas()

    input_texts_for_test = []
    masked_token_targets = []
    for _, cit in cit_df_for_test.iterrows():
        temp_masked_text = cit["masked_cit_context"]

        temp_masked_text = shorten_masked_context_for_limit_if_necessary(temp_masked_text)
        # Ignore lines that have been shortened too much (they have no mask)
        # --> Normally, this situation never happens.
        if temp_masked_text.find("<mask>") == -1:
            continue
        input_texts_for_test.append(temp_masked_text)

        masked_token_targets.append(cit['masked_token_target'])

    all_preds = mask_filler(input_texts_for_test)
    hit_count = 0
    pred_comparison_count = 0
    for j in range(len(all_preds)):
        pred_comparison_count += 1
        temp_preds = all_preds[j]

        target_pred_found = False
        for p in temp_preds:
            if isinstance(p, list):
                for p_in in p:
                    if p_in['token_str'] == masked_token_targets[j]:
                        hit_count += 1
                        target_pred_found = True
            elif p['token_str'] == masked_token_targets[j]:
                hit_count += 1
                target_pred_found = True

            if target_pred_found:
                break

    hit_at_k_metric = hit_count / pred_comparison_count
    print(f"\n=======>>> Hits@{k} score (between 0 and 1) = ", hit_at_k_metric, "\n")


def calc_exact_match_acc_score(val_dataset):
    mask_filler = pipeline(
        "fill-mask", model=model, tokenizer=tokenizer, top_k=3, device=0
    )
    cit_df_for_test = val_dataset.to_pandas()

    input_texts_for_test = []
    masked_token_targets = []
    for _, cit in cit_df_for_test.iterrows():
        temp_masked_text = cit["masked_cit_context"]

        temp_masked_text = shorten_masked_context_for_limit_if_necessary(temp_masked_text)
        if temp_masked_text.find("<mask>") == -1:
            continue
        input_texts_for_test.append(temp_masked_text)

        masked_token_targets.append(cit['masked_token_target'])

    all_preds = mask_filler(input_texts_for_test)
    exact_match_count = 0
    pred_comparison_count = 0
    for j in range(len(all_preds)):
        pred_comparison_count += 1
        temp_preds = all_preds[j]

        first_pred = temp_preds[0]
        if isinstance(first_pred, list):
            if first_pred[0]['token_str'] == masked_token_targets[j]:
                exact_match_count += 1
        elif first_pred['token_str'] == masked_token_targets[j]:
            exact_match_count += 1

    exact_match_metric = exact_match_count / pred_comparison_count
    print("\n=======>>> Exact match/accuracy score (between 0 and 1) = ", exact_match_metric, "\n")


def calc_mrr_score(val_dataset):
    mask_filler = pipeline(
        "fill-mask", model=model, tokenizer=tokenizer, top_k=20, device=0
    )
    cit_df_for_test = val_dataset.to_pandas()

    input_texts_for_test = []
    masked_token_targets = []
    for _, cit in cit_df_for_test.iterrows():
        temp_masked_text = cit["masked_cit_context"]

        temp_masked_text = shorten_masked_context_for_limit_if_necessary(temp_masked_text)
        if temp_masked_text.find("<mask>") == -1:
            continue
        input_texts_for_test.append(temp_masked_text)

        masked_token_targets.append(cit['masked_token_target'])

    all_preds = mask_filler(input_texts_for_test)
    temp_reciprocal_rank = 0
    reciprocal_rank_list = []

    for j in range(len(all_preds)):
        temp_preds = all_preds[j]
        reciprocal_rank_list.append(0)  # Start all recip ranks as 0. If match is found, then it is replaced.

        target_pred_found = False
        for p_idx in range(len(temp_preds)):
            if isinstance(temp_preds[p_idx], list):
                for p_in_idx in range(len(temp_preds[p_idx])):
                    if temp_preds[p_idx][p_in_idx]['token_str'] == masked_token_targets[j]:
                        temp_reciprocal_rank = 1 / (p_in_idx + 1)
                        target_pred_found = True
                        break
            elif temp_preds[p_idx]['token_str'] == masked_token_targets[j]:
                temp_reciprocal_rank = 1 / (p_idx + 1)
                target_pred_found = True

            if target_pred_found:
                # Replace 0 in the last index with the discovered RR value.
                reciprocal_rank_list[-1] = temp_reciprocal_rank
                break

    mean_reciprocal_rank = np.mean(reciprocal_rank_list)
    print("\n=======>>> MRR score = ", mean_reciprocal_rank, "\n")


def calc_recall_at_k_score(val_dataset, k=10):  # Since each example has only 1 ground truth, this is same as hits@10.
    mask_filler = pipeline(
        "fill-mask", model=model, tokenizer=tokenizer, top_k=k, device=0
    )
    cit_df_for_test = val_dataset.to_pandas()

    input_texts_for_test = []
    masked_token_targets = []
    for _, cit in cit_df_for_test.iterrows():
        temp_masked_text = cit["masked_cit_context"]

        temp_masked_text = shorten_masked_context_for_limit_if_necessary(temp_masked_text)
        if temp_masked_text.find("<mask>") == -1:
            continue
        input_texts_for_test.append(temp_masked_text)

        masked_token_targets.append(cit['masked_token_target'])

    all_preds = mask_filler(input_texts_for_test)

    recall_values_list = []
    total_num_of_relevant_items = 1  # Currently, there is only one relevant ground truth value per example.
    for j in range(len(all_preds)):
        temp_recall_value = 0
        temp_preds = all_preds[j]

        target_pred_found = False
        for p in temp_preds:
            if isinstance(p, list):
                for p_in in p:
                    if p_in['token_str'] == masked_token_targets[j]:
                        temp_recall_value += 1
                        target_pred_found = True
                        break
            elif p['token_str'] == masked_token_targets[j]:
                temp_recall_value += 1
                target_pred_found = True

            if target_pred_found:
                break
        recall_values_list.append(temp_recall_value / total_num_of_relevant_items)

    recall_at_k_score = np.mean(recall_values_list)
    print(f"\n=======>>> Recall@{k} score (between 0 and 1) = ", recall_at_k_score, "\n")


if __name__ == '__main__':
    tokenizer = RobertaTokenizer.from_pretrained(local_model_path, truncation=True, padding='max_length',
                                                 max_length=max_token_limit)
    model = RobertaForMaskedLM.from_pretrained(local_model_path)

    print("*** Added the new citations tokens to the tokenizer. Example for acl-200:\n",
          tokenizer.tokenize('Our paper is referencing the paper of Nenkova and Passonneau, 2004'), "\n\n")
    print("*** Another example for peerread:\n",
          tokenizer.tokenize('Our paper is referencing the paper of Gribkoff et al., 2014'), "\n\n")
    print("*** Another example for refseer:\n",
          tokenizer.tokenize('Our paper is referencing the paper of Lecoutre and Boussemart, 2003'), "\n\n")
    print("*** Another example for arxiv:\n",
          tokenizer.tokenize('Our paper is referencing the paper of Fishman et al., 2009'), "\n\n")

    eval_dataset = read_eval_dataset(tokenizer)

    print("~" * 40)
    print("\n*** Calculating Hits@10 score")
    calc_hits_at_k_score(eval_dataset, k=10)

    print("~" * 40)
    print("\n*** Calculating Exact Match/Accuracy score")
    calc_exact_match_acc_score(eval_dataset)

    print("~" * 40)
    print("\n*** Calculating MRR score")
    calc_mrr_score(eval_dataset)

    print("~" * 40)
    print("\n*** Calculating Recall@10 score")
    calc_recall_at_k_score(eval_dataset, k=10)
