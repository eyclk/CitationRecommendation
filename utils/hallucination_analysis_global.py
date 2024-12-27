from typing import List, Any
# from datasets import DatasetDict, Dataset
from transformers import (BartForConditionalGeneration, BartTokenizer,
                          BartConfig, GenerationConfig)
# from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments
import pandas as pd
import argparse
# import math
from tqdm import tqdm
#     import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("--max_token_limit", type=int, default=400, help="Max amount allowed for tokens used for training "
                                                                     "and evaluation")
parser.add_argument("--model_name", type=str, help="The name of the new model. This is for saved model and checkpoints")
parser.add_argument("--checkpoints_path", type=str, default="../checkpoints", help="Path of the checkpoints folder")
parser.add_argument("--models_path", type=str, default="../models", help="Path of the models folder")
parser.add_argument("--dataset_path", type=str, help="Path to the folder of the dataset")
parser.add_argument("--num_epochs", type=int, default=5, help="Number of epochs for training")
parser.add_argument("--warmup_steps", type=int, default=500, help="Number of warmup steps for the learning rate")
parser.add_argument("--batch_size", type=int, default=16, help="Batch size for the training and evaluation")
parser.add_argument("--pretrained_model_path", type=str, default="facebook/bart-base", help="Path or name "
                                                                                            "of the pretrained model "
                                                                                            "used at the beginning")
parser.add_argument("--auto_find_batch_size", type=bool, default=False, help="Make this flag True for the Trainer to "
                                                                             "automatically select an appropriate "
                                                                             "batch size")
parser.add_argument("--skip_training", type=bool, default=False, help="Skips training and directly perform evaluation")


# Preprocessing function
def preprocess_function(examples):
    inputs = [example.replace("<mask>", "<extra_id_0>", 1).replace("<mask>", " ").replace("<extra_id_0>", "<mask>")
              for example in examples["masked_cit_context"]]
    targets = [example for example in examples["masked_token_target"]]

    model_inputs = tokenizer(inputs, max_length=max_token_limit, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=max_token_limit, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def read_dataset():
    train_df = pd.read_csv(train_dataset_path)  # , nrows=300
    train_set = []

    for _, i in train_df.iterrows():
        temp_citing_title = i['citing_title']
        temp_citing_abstract = i['citing_abstract']
        temp_masked_context = i['masked_cit_context'].replace("OTHERCIT", "")

        temp_train_input = temp_citing_title + " </s> " + temp_citing_abstract + " </s> " + temp_masked_context

        temp_dict = {"masked_cit_context": temp_train_input,
                     "masked_token_target": i['masked_token_target']}

        train_set.append(temp_dict)

    eval_df = pd.read_csv(eval_dataset_path)  # , nrows=300
    eval_set = []

    for _, i in eval_df.iterrows():
        temp_citing_title = i['citing_title']
        temp_citing_abstract = i['citing_abstract']
        temp_masked_context = i['masked_cit_context'].replace("OTHERCIT", "")

        temp_eval_input = temp_citing_title + " </s> " + temp_citing_abstract + " </s> " + temp_masked_context

        temp_dict = {"masked_cit_context": temp_eval_input,
                     "masked_token_target": i['masked_token_target']}

        eval_set.append(temp_dict)

    return train_set, eval_set


def add_spaces_after_commas(text: str) -> str:
    return ', '.join(part.strip() for part in text.split(','))


def fill_mask(sentence):
    input_ids = tokenizer.encode(sentence.replace("<mask>", "<extra_id_0>").replace("<mask>", "").
                                 replace("<extra_id_0>", "<mask>"),
                                 return_tensors="pt", max_length=max_token_limit, truncation=True,
                                 padding="max_length").to("cuda")

    # model.to("cuda")

    outputs = model.generate(
        input_ids,
        generation_config=cit_generation_config
    )

    predictions = []
    for output in outputs:
        decoded_output = tokenizer.decode(output, skip_special_tokens=True)
        temp_prediction = decoded_output.strip()

        temp_prediction = add_spaces_after_commas(temp_prediction)  # Add spaces after commas

        predictions.append(temp_prediction)

    # Get unique predictions
    unique_predictions: List[Any] = list(dict.fromkeys(predictions))  # Remove duplicates while preserving order

    # Print the top 10 predictions
    # for i, pred in enumerate(unique_predictions, 1):
    #     print(f"Prediction {i}: {pred} \n\n")

    last_item_of_predictions = unique_predictions[-1]
    while len(unique_predictions) < 10:
        unique_predictions.append(last_item_of_predictions)

    # Return the top 10 unique predictions, or fewer if not enough unique ones are available
    return unique_predictions


def compare_pred_with_correct_value(predictions, ground_truth, top_k=10):
    hits_at_k_flag = {10: False, 5: False, 3: False}
    exact_match_flag = False
    temp_reciprocal_rank = 0

    if "and" in ground_truth:
        truth_tokens = ground_truth.replace(" and ", ", ").replace(",", "").split()
        if len(truth_tokens) == 3:
            for p_idx in range(top_k):  # len(predictions)
                if (truth_tokens[0] in predictions[p_idx] and truth_tokens[1] in predictions[p_idx] and
                        truth_tokens[2] in predictions[p_idx]):
                    if p_idx < 3:
                        hits_at_k_flag[10] = True
                        hits_at_k_flag[5] = True
                        hits_at_k_flag[3] = True
                    elif p_idx < 5:
                        hits_at_k_flag[10] = True
                        hits_at_k_flag[5] = True
                    elif p_idx < 10:
                        hits_at_k_flag[10] = True

                    temp_reciprocal_rank = 1 / (p_idx + 1)
                    break
            if (truth_tokens[0] in predictions[0] and truth_tokens[1] in predictions[0] and
                    truth_tokens[2] in predictions[0]):
                exact_match_flag = True

    elif "et al" in ground_truth:
        truth_tokens = ground_truth.replace(" et al.,", "").split()
        for p_idx in range(top_k):
            if truth_tokens[0] in predictions[p_idx] and truth_tokens[1] in predictions[p_idx]:
                if p_idx < 3:
                    hits_at_k_flag[10] = True
                    hits_at_k_flag[5] = True
                    hits_at_k_flag[3] = True
                elif p_idx < 5:
                    hits_at_k_flag[10] = True
                    hits_at_k_flag[5] = True
                elif p_idx < 10:
                    hits_at_k_flag[10] = True

                temp_reciprocal_rank = 1 / (p_idx + 1)
                break
        if truth_tokens[0] in predictions[0] and truth_tokens[1] in predictions[0]:
            exact_match_flag = True
    else:
        truth_tokens = ground_truth.replace(",", "").split()
        for p_idx in range(top_k):
            if truth_tokens[0] in predictions[p_idx] and truth_tokens[1] in predictions[p_idx]:
                if p_idx < 3:
                    hits_at_k_flag[10] = True
                    hits_at_k_flag[5] = True
                    hits_at_k_flag[3] = True
                elif p_idx < 5:
                    hits_at_k_flag[10] = True
                    hits_at_k_flag[5] = True
                elif p_idx < 10:
                    hits_at_k_flag[10] = True

                temp_reciprocal_rank = 1 / (p_idx + 1)
                break
        if truth_tokens[0] in predictions[0] and truth_tokens[1] in predictions[0]:
            exact_match_flag = True

    # if hits_at_k_flag[10] is False:
    for p_idx in range(top_k):
        if predictions[p_idx] == ground_truth:
            if p_idx < 3:
                hits_at_k_flag[10] = True
                hits_at_k_flag[5] = True
                hits_at_k_flag[3] = True
            elif p_idx < 5:
                hits_at_k_flag[10] = True
                hits_at_k_flag[5] = True
            elif p_idx < 10:
                hits_at_k_flag[10] = True
            # temp_reciprocal_rank = 1 / (p_idx + 1)
            break

    if predictions[0] == ground_truth:
        exact_match_flag = True

    return hits_at_k_flag, exact_match_flag, temp_reciprocal_rank


def check_if_word_is_hallucinated(word):
    no_hal_flag = False
    for c in all_cit_list:
        if word in c:
            no_hal_flag = True
    return not no_hal_flag


def find_hallucination_rates_for_top_10_5_3(predictions, ground_truth, top_k=10):
    fabricated_word_hal_count = {10: 0, 5: 0, 3: 0}

    for p_id in range(top_k):
        temp_pred = predictions[p_id]
        if temp_pred not in all_cit_list:
            if "and" in temp_pred:
                pred_tokens = temp_pred.replace(" and ", ", ").replace(",", "").split()
                if len(pred_tokens) == 3 and (check_if_word_is_hallucinated(pred_tokens[0]) or
                                              check_if_word_is_hallucinated(pred_tokens[1])):
                    if p_id < 3:
                        fabricated_word_hal_count[10] += 1
                        fabricated_word_hal_count[5] += 1
                        fabricated_word_hal_count[3] += 1
                    elif p_id < 5:
                        fabricated_word_hal_count[10] += 1
                        fabricated_word_hal_count[5] += 1
                    elif p_id < 10:
                        fabricated_word_hal_count[10] += 1
            elif "et al" in temp_pred:
                pred_tokens = temp_pred.replace(" et al.,", "").split()
                if check_if_word_is_hallucinated(pred_tokens[0]):
                    if p_id < 3:
                        fabricated_word_hal_count[10] += 1
                        fabricated_word_hal_count[5] += 1
                        fabricated_word_hal_count[3] += 1
                    elif p_id < 5:
                        fabricated_word_hal_count[10] += 1
                        fabricated_word_hal_count[5] += 1
                    elif p_id < 10:
                        fabricated_word_hal_count[10] += 1
            else:
                pred_tokens = temp_pred.replace(",", "").split()
                if check_if_word_is_hallucinated(pred_tokens[0]):
                    if p_id < 3:
                        fabricated_word_hal_count[10] += 1
                        fabricated_word_hal_count[5] += 1
                        fabricated_word_hal_count[3] += 1
                    elif p_id < 5:
                        fabricated_word_hal_count[10] += 1
                        fabricated_word_hal_count[5] += 1
                    elif p_id < 10:
                        fabricated_word_hal_count[10] += 1

    hallucination_count = {10: 0, 5: 0, 3: 0}
    only_author_names_correct_count = {10: 0, 5: 0, 3: 0}
    only_year_correct_count = {10: 0, 5: 0, 3: 0}
    wrong_cite_format_count = {10: 0, 5: 0, 3: 0}
    only_single_author_name_correct_count = {10: 0, 5: 0, 3: 0}

    if "and" in ground_truth:
        truth_tokens = ground_truth.replace(" and ", ", ").replace(",", "").split()
        if len(truth_tokens) == 3:
            for p_idx in range(top_k):
                if predictions[p_idx] not in all_cit_list:
                    # print(f"Prediction {p_idx+1}: {predictions[p_idx]}")  # TEMP
                    if p_idx < 3:
                        hallucination_count[10] += 1
                        hallucination_count[5] += 1
                        hallucination_count[3] += 1
                    elif p_idx < 5:
                        hallucination_count[10] += 1
                        hallucination_count[5] += 1
                    elif p_idx < 10:
                        hallucination_count[10] += 1

                    if (truth_tokens[0] in predictions[p_idx] and truth_tokens[1] in predictions[p_idx]
                            and truth_tokens[2] not in predictions[p_idx]):
                        if p_idx < 3:
                            only_author_names_correct_count[10] += 1
                            only_author_names_correct_count[5] += 1
                            only_author_names_correct_count[3] += 1
                        elif p_idx < 5:
                            only_author_names_correct_count[10] += 1
                            only_author_names_correct_count[5] += 1
                        elif p_idx < 10:
                            only_author_names_correct_count[10] += 1

                    elif (truth_tokens[0] in predictions[p_idx] or truth_tokens[1] in predictions[p_idx]
                            and truth_tokens[2] not in predictions[p_idx]):
                        if p_idx < 3:
                            only_single_author_name_correct_count[10] += 1
                            only_single_author_name_correct_count[5] += 1
                            only_single_author_name_correct_count[3] += 1
                        elif p_idx < 5:
                            only_single_author_name_correct_count[10] += 1
                            only_single_author_name_correct_count[5] += 1
                        elif p_idx < 10:
                            only_single_author_name_correct_count[10] += 1

                    elif (truth_tokens[0] not in predictions[p_idx] and truth_tokens[1] not in predictions[p_idx]
                            and truth_tokens[2] in predictions[p_idx]):
                        if p_idx < 3:
                            only_year_correct_count[10] += 1
                            only_year_correct_count[5] += 1
                            only_year_correct_count[3] += 1
                        elif p_idx < 5:
                            only_year_correct_count[10] += 1
                            only_year_correct_count[5] += 1
                        elif p_idx < 10:
                            only_year_correct_count[10] += 1

                    elif (" and " in predictions[p_idx] and ".," in predictions[p_idx]) or "&" in predictions[p_idx]:
                        if p_idx < 3:
                            wrong_cite_format_count[10] += 1
                            wrong_cite_format_count[5] += 1
                            wrong_cite_format_count[3] += 1
                        elif p_idx < 5:
                            wrong_cite_format_count[10] += 1
                            wrong_cite_format_count[5] += 1
                        elif p_idx < 10:
                            wrong_cite_format_count[10] += 1

    elif "et al" in ground_truth:
        truth_tokens = ground_truth.replace(" et al.,", "").split()
        for p_idx in range(top_k):
            if predictions[p_idx] not in all_cit_list:
                # print(f"Prediction {p_idx + 1}: {predictions[p_idx]}")  # TEMP
                if p_idx < 3:
                    hallucination_count[10] += 1
                    hallucination_count[5] += 1
                    hallucination_count[3] += 1
                elif p_idx < 5:
                    hallucination_count[10] += 1
                    hallucination_count[5] += 1
                elif p_idx < 10:
                    hallucination_count[10] += 1

                if truth_tokens[0] in predictions[p_idx] and truth_tokens[1] not in predictions[p_idx]:
                    if p_idx < 3:
                        only_author_names_correct_count[10] += 1
                        only_author_names_correct_count[5] += 1
                        only_author_names_correct_count[3] += 1
                    elif p_idx < 5:
                        only_author_names_correct_count[10] += 1
                        only_author_names_correct_count[5] += 1
                    elif p_idx < 10:
                        only_author_names_correct_count[10] += 1

                elif truth_tokens[0] not in predictions[p_idx] and truth_tokens[1] in predictions[p_idx]:
                    if p_idx < 3:
                        only_year_correct_count[10] += 1
                        only_year_correct_count[5] += 1
                        only_year_correct_count[3] += 1
                    elif p_idx < 5:
                        only_year_correct_count[10] += 1
                        only_year_correct_count[5] += 1
                    elif p_idx < 10:
                        only_year_correct_count[10] += 1

                elif (" and " in predictions[p_idx] and ".," in predictions[p_idx]) or "&" in predictions[p_idx]:
                    if p_idx < 3:
                        wrong_cite_format_count[10] += 1
                        wrong_cite_format_count[5] += 1
                        wrong_cite_format_count[3] += 1
                    elif p_idx < 5:
                        wrong_cite_format_count[10] += 1
                        wrong_cite_format_count[5] += 1
                    elif p_idx < 10:
                        wrong_cite_format_count[10] += 1
    else:
        truth_tokens = ground_truth.replace(",", "").split()
        for p_idx in range(top_k):
            if predictions[p_idx] not in all_cit_list:
                # print(f"Prediction {p_idx + 1}: {predictions[p_idx]}")  # TEMP
                if p_idx < 3:
                    hallucination_count[10] += 1
                    hallucination_count[5] += 1
                    hallucination_count[3] += 1
                elif p_idx < 5:
                    hallucination_count[10] += 1
                    hallucination_count[5] += 1
                elif p_idx < 10:
                    hallucination_count[10] += 1

                if truth_tokens[0] in predictions[p_idx] and truth_tokens[1] not in predictions[p_idx]:
                    if p_idx < 3:
                        only_author_names_correct_count[10] += 1
                        only_author_names_correct_count[5] += 1
                        only_author_names_correct_count[3] += 1
                    elif p_idx < 5:
                        only_author_names_correct_count[10] += 1
                        only_author_names_correct_count[5] += 1
                    elif p_idx < 10:
                        only_author_names_correct_count[10] += 1

                elif truth_tokens[0] not in predictions[p_idx] and truth_tokens[1] in predictions[p_idx]:
                    if p_idx < 3:
                        only_year_correct_count[10] += 1
                        only_year_correct_count[5] += 1
                        only_year_correct_count[3] += 1
                    elif p_idx < 5:
                        only_year_correct_count[10] += 1
                        only_year_correct_count[5] += 1
                    elif p_idx < 10:
                        only_year_correct_count[10] += 1

                elif (" and " in predictions[p_idx] and ".," in predictions[p_idx]) or "&" in predictions[p_idx]:
                    if p_idx < 3:
                        wrong_cite_format_count[10] += 1
                        wrong_cite_format_count[5] += 1
                        wrong_cite_format_count[3] += 1
                    elif p_idx < 5:
                        wrong_cite_format_count[10] += 1
                        wrong_cite_format_count[5] += 1
                    elif p_idx < 10:
                        wrong_cite_format_count[10] += 1

    return hallucination_count, only_author_names_correct_count, only_year_correct_count, wrong_cite_format_count, fabricated_word_hal_count, only_single_author_name_correct_count


def calc_eval_metrics(val_dataset, top_k=10):
    exact_match_count = 0
    reciprocal_rank_list = []
    pred_comparison_count = 0

    total_hal_count = {10: 0, 5: 0, 3: 0}
    total_only_author_names_correct_count = {10: 0, 5: 0, 3: 0}
    total_only_year_correct_count = {10: 0, 5: 0, 3: 0}
    total_incorrect_format_count = {10: 0, 5: 0, 3: 0}
    total_fabricated_word_hal_count = {10: 0, 5: 0, 3: 0}
    total_single_author_name_correct_count = {10: 0, 5: 0, 3: 0}

    if_hit_at_k_hal_count = {10: 0, 5: 0, 3: 0}
    if_exact_match_hal_count = {10: 0, 5: 0, 3: 0}

    for e in tqdm(val_dataset):
        pred_comparison_count += 1
        masked_cit_context = e["masked_cit_context"]
        target_token = e["masked_token_target"]

        temp_predictions = fill_mask(masked_cit_context)
        # print(f"\n--> Ground truth cit = {target_token}\n\n")
        hits_at_k_flag, exact_match_flag, temp_reciprocal_rank = compare_pred_with_correct_value(temp_predictions,
                                                                                                  target_token, top_k=top_k)

        (hallucination_count, only_author_names_correct_count, only_year_correct_count, incorrect_format_count, fabricated_word_hal_count,
         only_single_author_name_correct_count) = find_hallucination_rates_for_top_10_5_3(temp_predictions, target_token, top_k=top_k)

        total_hal_count[10] += hallucination_count[10]
        total_hal_count[5] += hallucination_count[5]
        total_hal_count[3] += hallucination_count[3]

        total_only_author_names_correct_count[10] += only_author_names_correct_count[10]
        total_only_author_names_correct_count[5] += only_author_names_correct_count[5]
        total_only_author_names_correct_count[3] += only_author_names_correct_count[3]

        total_only_year_correct_count[10] += only_year_correct_count[10]
        total_only_year_correct_count[5] += only_year_correct_count[5]
        total_only_year_correct_count[3] += only_year_correct_count[3]

        total_incorrect_format_count[10] += incorrect_format_count[10]
        total_incorrect_format_count[5] += incorrect_format_count[5]
        total_incorrect_format_count[3] += incorrect_format_count[3]

        total_fabricated_word_hal_count[10] += fabricated_word_hal_count[10]
        total_fabricated_word_hal_count[5] += fabricated_word_hal_count[5]
        total_fabricated_word_hal_count[3] += fabricated_word_hal_count[3]

        total_single_author_name_correct_count[10] += only_single_author_name_correct_count[10]
        total_single_author_name_correct_count[5] += only_single_author_name_correct_count[5]
        total_single_author_name_correct_count[3] += only_single_author_name_correct_count[3]

        if hits_at_k_flag[3]:
            if_hit_at_k_hal_count[10] += hallucination_count[10]
            if_hit_at_k_hal_count[5] += hallucination_count[5]
            if_hit_at_k_hal_count[3] += hallucination_count[3]
        elif hits_at_k_flag[5]:
            if_hit_at_k_hal_count[10] += hallucination_count[10]
            if_hit_at_k_hal_count[5] += hallucination_count[5]
        elif hits_at_k_flag[10]:
            if_hit_at_k_hal_count[10] += hallucination_count[10]

        if exact_match_flag:
            if_exact_match_hal_count[10] += hallucination_count[10]
            if_exact_match_hal_count[5] += hallucination_count[5]
            if_exact_match_hal_count[3] += hallucination_count[3]

        if exact_match_flag:
            exact_match_count += 1
        reciprocal_rank_list.append(temp_reciprocal_rank)

    #     hit_at_10_metric = hit_count / pred_comparison_count
    #     exact_match_metric = exact_match_count / pred_comparison_count
    #     print("\n=======>>> Exact match (accuracy) measurement value (between 0 and 1) = ", exact_match_metric, "\n")
    #     mean_reciprocal_rank = np.mean(reciprocal_rank_list)
    #     print("\n=======>>> MRR score value = ", mean_reciprocal_rank, "\n")
    #     print("\n=======>>> Recall@10 measurement value (between 0 and 1) = ", hit_at_10_metric, "\n")


    print("\n\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ TOP-10 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")

    print(f"\n=======>>> Total number of predictions in top-10 case = {pred_comparison_count * 10}\n")

    hal_rate_10 = total_hal_count[10] / (pred_comparison_count * 10)
    print("\n=======>>> Hallucination rate (any prediction that does not belong to all citations list of the dataset is considered to be hallucination) = ", hal_rate_10, "\n")
    # print(f"\n=======>>> Total number of hallucinated predictions = {total_hal_count} in {pred_comparison_count} * 10\n")  #TEMP

    partial_correct_rate_10 = (total_only_author_names_correct_count[10]+total_single_author_name_correct_count[10]+total_only_year_correct_count[10]) / (pred_comparison_count * 10)
    print("\n=======>>> Partially correct predictions rate among hallucinated predictions = ", partial_correct_rate_10, "\n")
    only_author_names_correct_rate_10 = total_only_author_names_correct_count[10] / (pred_comparison_count * 10)
    print("\n=======>>> Rate of hallucinated predictions with only their author names correct (in \"and\" case, both names should be correct) = ", only_author_names_correct_rate_10, "\n")
    total_single_author_name_correct_count_rate_10 = total_single_author_name_correct_count[10] / (pred_comparison_count * 10)
    print("\n=======>>> Rate of hallucinated predictions with only one of the author names correct in \"and\" case = ", total_single_author_name_correct_count_rate_10, "\n")

    only_year_correct_rate_10 = total_only_year_correct_count[10] / (pred_comparison_count * 10)
    print("\n=======>>> Rate of hallucinated predictions with only their publication years correct = ", only_year_correct_rate_10, "\n")
    incorrect_format_rate_10 = total_incorrect_format_count[10] / (pred_comparison_count * 10)
    print("\n=======>>> Rate of hallucinated predictions with incorrect citation format = ", incorrect_format_rate_10, "\n")
    other_hallucinations_rate_10 = (total_hal_count[10] - total_only_author_names_correct_count[10] - total_single_author_name_correct_count[10] -
                                    total_only_year_correct_count[10] - total_incorrect_format_count[10]) / (pred_comparison_count * 10)
    print("\n=======>>> Rate of hallucinated predictions with other hallucinations (Correct citation format with irrelevant "
          "author-year combinations that has no matches with ground truth) = ", other_hallucinations_rate_10, "\n")

    fabricated_word_hal_rate_10 = total_fabricated_word_hal_count[10] / (pred_comparison_count * 10)
    print("\n=======>>> Rate of hallucinated predictions with fabricated words (independent rate from other rates) = ", fabricated_word_hal_rate_10, "\n")

    if_hit_at_k_hal_rate_10 = if_hit_at_k_hal_count[10] / (pred_comparison_count * 10)
    print(f"\n=======>>> Rate of all hallucinated predictions in top-10 in cases where there is a hit = {if_hit_at_k_hal_rate_10}\n")

    if_exact_match_hal_rate_10 = if_exact_match_hal_count[10] / (pred_comparison_count * 10)
    print(f"\n=======>>> Rate of all hallucinated predictions in top-10 in cases where there is an exact match = {if_exact_match_hal_rate_10}\n")


    print("\n\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ TOP-5 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")

    print(f"\n=======>>> Total number of predictions in top-5 case = {pred_comparison_count * 5}\n")

    hal_rate_5 = total_hal_count[5] / (pred_comparison_count * 5)
    print("\n=======>>> Hallucination rate (any prediction that does not belong to all citations list of the dataset is considered to be hallucination) = ", hal_rate_5, "\n")

    partial_correct_rate_5 = (total_only_author_names_correct_count[5]+total_single_author_name_correct_count[5]+total_only_year_correct_count[5]) / (pred_comparison_count * 5)
    print("\n=======>>> Partially correct predictions rate among hallucinated predictions = ", partial_correct_rate_5, "\n")
    only_author_names_correct_rate_5 = total_only_author_names_correct_count[5] / (pred_comparison_count * 5)
    print("\n=======>>> Rate of hallucinated predictions with only their author names correct (in \"and\" case, both names should be correct) = ", only_author_names_correct_rate_5, "\n")
    total_single_author_name_correct_count_rate_5 = total_single_author_name_correct_count[5] / (pred_comparison_count * 5)
    print("\n=======>>> Rate of hallucinated predictions with only one of the author names correct in \"and\" case = ", total_single_author_name_correct_count_rate_5, "\n")

    only_year_correct_rate_5 = total_only_year_correct_count[5] / (pred_comparison_count * 5)
    print("\n=======>>> Rate of hallucinated predictions with only their publication years correct = ", only_year_correct_rate_5, "\n")
    incorrect_format_rate_5 = total_incorrect_format_count[5] / (pred_comparison_count * 5)
    print("\n=======>>> Rate of hallucinated predictions with incorrect citation format = ", incorrect_format_rate_5, "\n")
    other_hallucinations_rate_5 = (total_hal_count[5] - total_only_author_names_correct_count[5] - total_single_author_name_correct_count[5] -
                                      total_only_year_correct_count[5] - total_incorrect_format_count[5]) / (pred_comparison_count * 5)
    print("\n=======>>> Rate of hallucinated predictions with other hallucinations (Correct citation format with irrelevant "
            "author-year combinations that has no matches with ground truth) = ", other_hallucinations_rate_5, "\n")

    fabricated_word_hal_rate_5 = total_fabricated_word_hal_count[5] / (pred_comparison_count * 5)
    print("\n=======>>> Rate of hallucinated predictions with fabricated words (independent rate from other rates) = ", fabricated_word_hal_rate_5, "\n")

    if_hit_at_k_hal_rate_5 = if_hit_at_k_hal_count[5] / (pred_comparison_count * 5)
    print(f"\n=======>>> Rate of all hallucinated predictions in top-5 in cases where there is a hit = {if_hit_at_k_hal_rate_5}\n")

    if_exact_match_hal_rate_5 = if_exact_match_hal_count[5] / (pred_comparison_count * 5)
    print(f"\n=======>>> Rate of all hallucinated predictions in top-5 in cases where there is an exact match = {if_exact_match_hal_rate_5}\n")


    print("\n\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ TOP-3 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")

    print(f"\n=======>>> Total number of predictions in top-3 case = {pred_comparison_count * 3}\n")

    hal_rate_3 = total_hal_count[3] / (pred_comparison_count * 3)
    print("\n=======>>> Hallucination rate (any prediction that does not belong to all citations list of the dataset is considered to be hallucination) = ", hal_rate_3, "\n")

    partial_correct_rate_3 = (total_only_author_names_correct_count[3]+total_single_author_name_correct_count[3]+total_only_year_correct_count[3]) / (pred_comparison_count * 3)
    print("\n=======>>> Partially correct predictions rate among hallucinated predictions = ", partial_correct_rate_3, "\n")
    only_author_names_correct_rate_3 = total_only_author_names_correct_count[3] / (pred_comparison_count * 3)
    print("\n=======>>> Rate of hallucinated predictions with only their author names correct (in \"and\" case, both names should be correct) = ", only_author_names_correct_rate_3, "\n")
    total_single_author_name_correct_count_rate_3 = total_single_author_name_correct_count[3] / (pred_comparison_count * 3)
    print("\n=======>>> Rate of hallucinated predictions with only one of the author names correct in \"and\" case = ", total_single_author_name_correct_count_rate_3, "\n")

    only_year_correct_rate_3 = total_only_year_correct_count[3] / (pred_comparison_count * 3)
    print("\n=======>>> Rate of hallucinated predictions with only their publication years correct = ", only_year_correct_rate_3, "\n")
    incorrect_format_rate_3 = total_incorrect_format_count[3] / (pred_comparison_count * 3)
    print("\n=======>>> Rate of hallucinated predictions with incorrect citation format = ", incorrect_format_rate_3, "\n")
    other_hallucinations_rate_3 = (total_hal_count[3] - total_only_author_names_correct_count[3] - total_single_author_name_correct_count[3] -
                                        total_only_year_correct_count[3] - total_incorrect_format_count[3]) / (pred_comparison_count * 3)
    print("\n=======>>> Rate of hallucinated predictions with other hallucinations (Correct citation format with irrelevant "
            "author-year combinations that has no matches with ground truth) = ", other_hallucinations_rate_3, "\n")

    fabricated_word_hal_rate_3 = total_fabricated_word_hal_count[3] / (pred_comparison_count * 3)
    print("\n=======>>> Rate of hallucinated predictions with fabricated words (independent rate from other rates) = ", fabricated_word_hal_rate_3, "\n")

    if_hit_at_k_hal_rate_3 = if_hit_at_k_hal_count[3] / (pred_comparison_count * 3)
    print(f"\n=======>>> Rate of all hallucinated predictions in top-3 in cases where there is a hit = {if_hit_at_k_hal_rate_3}\n")

    if_exact_match_hal_rate_3 = if_exact_match_hal_count[3] / (pred_comparison_count * 3)
    print(f"\n=======>>> Rate of all hallucinated predictions in top-3 in cases where there is an exact match = {if_exact_match_hal_rate_3}\n")


if __name__ == '__main__':
    args = parser.parse_args()

    max_token_limit = args.max_token_limit
    custom_model_name = args.model_name
    checkpoints_location = f"{args.checkpoints_path}/{custom_model_name}"
    model_save_location = f"{args.models_path}/{custom_model_name}"

    dataset_folder = args.dataset_path
    train_dataset_path = dataset_folder + "/context_dataset_train.csv"
    eval_dataset_path = dataset_folder + "/context_dataset_eval.csv"

    all_citations_path = dataset_folder + "/citation_item_list.csv"

    num_epochs = args.num_epochs

    warmup_steps = args.warmup_steps
    train_and_eval_batch_sizes = args.batch_size

    auto_find_batch_size_flag = args.auto_find_batch_size

    pretrained_model_name_or_path = args.pretrained_model_path

    skip_training = args.skip_training

    # Initialize the config
    config = BartConfig.from_pretrained(pretrained_model_name_or_path, attention_dropout=0.123)

    # Initialize the tokenizer
    tokenizer = BartTokenizer.from_pretrained(pretrained_model_name_or_path, truncation=True,
                                              padding='max_length', model_max_length=max_token_limit)

    # Set up the model
    model = BartForConditionalGeneration.from_pretrained(pretrained_model_name_or_path, config=config).to("cuda")

    cit_generation_config = GenerationConfig.from_model_config(model.config)

    cit_generation_config.max_new_tokens = 25
    cit_generation_config.do_sample = False
    cit_generation_config.top_k = 50
    cit_generation_config.num_return_sequences = 20
    cit_generation_config.early_stopping = False
    cit_generation_config.num_beams = 20
    cit_generation_config.forced_bos_token_id = 0

    cit_generation_config.num_beam_groups = 10
    cit_generation_config.diversity_penalty = 1.5

    train_dataset, eval_dataset = read_dataset()

    """data = {
        "train": train_dataset,
        "eval": eval_dataset
    }

    # Convert to Dataset
    train_dataset = Dataset.from_pandas(pd.DataFrame(data["train"]))
    validation_dataset = Dataset.from_pandas(pd.DataFrame(data["eval"]))

    dataset = DatasetDict({
        "train": train_dataset,
        "eval": validation_dataset
    })

    # Preprocess the datasets
    tokenized_datasets = dataset.map(preprocess_function, batched=True)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    training_args = TrainingArguments(
        output_dir=checkpoints_location,
        overwrite_output_dir=True,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        logging_strategy="epoch",
        warmup_steps=warmup_steps,
        save_strategy="epoch",
        save_total_limit=5
    )

    if auto_find_batch_size_flag is True:
        training_args.auto_find_batch_size = True
    else:
        training_args.per_device_train_batch_size = train_and_eval_batch_sizes
        training_args.per_device_eval_batch_size = train_and_eval_batch_sizes

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["eval"],
        data_collator=data_collator,
        tokenizer=tokenizer
    )

    if not skip_training:
        trainer.train()

        trainer.save_model(model_save_location)
        tokenizer.save_pretrained(model_save_location)

    eval_results = trainer.evaluate()
    print(f"\n*****************\n======>> Eval loss after fine-tuning: {eval_results['eval_loss']}\n"
          f"======>> Perplexity after fine-tuning: {math.exp(eval_results['eval_loss']):.2f}\n\n")"""

    all_cit_df = pd.read_csv(all_citations_path)

    all_cit_list = []
    for _, n in all_cit_df.iterrows():
        temp_cit = n['citation_items']
        all_cit_list.append(temp_cit)


    calc_eval_metrics(eval_dataset, top_k=10)
