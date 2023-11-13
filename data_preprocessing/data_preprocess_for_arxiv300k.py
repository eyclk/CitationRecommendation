import pandas as pd
import random
from transformers import RobertaTokenizer
import json

contexts_file = "./arxiv_original/contexts.json"
papers_file = "./arxiv_original/papers.json"

dataset_output_file = "./arxiv300k_select_best_300k/context_dataset.csv"
vocab_output_file = "./arxiv300k_select_best_300k/additions_to_vocab.csv"
train_set_output_file = "./arxiv300k_select_best_300k/context_dataset_train.csv"
eval_set_output_file = "./arxiv300k_select_best_300k/context_dataset_eval.csv"

max_dataset_size = 300000
random.seed(42)

appearance_count_per_cit_dict_file = "../dataset_statistic_files/arxiv/appearance_count_per_cit_name_dict.json"


def assign_appropriate_year_for_null_years(ref_id, author_names):
    if ref_id not in dict_missing_years_for_refid.keys():
        random_year = random.randint(1991, 2020)
        year_names_tuple = [random_year, author_names]

        repeat_flag = True
        while repeat_flag:
            for k in dict_missing_years_for_refid:
                if dict_missing_years_for_refid[k] == year_names_tuple:
                    random_year = random.randint(1960, 2014)
                    year_names_tuple = [random_year, author_names]
                    break
            repeat_flag = False

        dict_missing_years_for_refid[ref_id] = [random_year, author_names]
    else:
        random_year = dict_missing_years_for_refid[ref_id][0]

    return str(random_year)


dict_missing_years_for_refid = {}


def create_target_token_for_ref_paper_id(ref_id, papers_df):
    target_cit_token = ""
    temp_paper_info_row = papers_df[int(ref_id)]
    authors_from_paper_info = temp_paper_info_row['authors']

    if temp_paper_info_row['year'] == 'NULL':  # THERE IS NO NULL YEAR IN ARXIV !!!!
        year_from_paper_info = assign_appropriate_year_for_null_years(ref_id, authors_from_paper_info)
    else:
        year_from_paper_info = str(int(float(temp_paper_info_row['year'])))

    if len(authors_from_paper_info) == 1:
        target_cit_token = authors_from_paper_info[0].split(" ")[-1].capitalize() + ", " + year_from_paper_info
    elif len(authors_from_paper_info) == 2:
        target_cit_token = authors_from_paper_info[0].split(" ")[-1].capitalize() + " and " + \
                           authors_from_paper_info[1].split(" ")[-1].capitalize() + ", " + year_from_paper_info
    elif len(authors_from_paper_info) > 2:
        target_cit_token = authors_from_paper_info[0].split(" ")[-1].capitalize() + " et al., " + year_from_paper_info

    return target_cit_token


def create_300k_dataset_standard():
    contexts_df = pd.read_json(contexts_file)
    papers_df = pd.read_json(papers_file)

    cit_contexts_list = []
    masked_cit_contexts_list = []
    masked_token_target_list = []

    skip_count = 0
    context_df_length = len(contexts_df.columns)

    for i in range(context_df_length):
        temp_context_row = contexts_df.iloc[:, i]

        # For arxiv; I have to use 'refid' values similar to peerread!!!
        temp_target_token = create_target_token_for_ref_paper_id(temp_context_row['refid'], papers_df)
        # If author names are invalid, function above will return empty string. But this never happens.
        if temp_target_token == "":
            skip_count += 1
            continue

        temp_masked_text = temp_context_row['masked_text']
        temp_masked_text = temp_masked_text.replace('OTHERCIT', '')

        masked_with_mask_text = temp_masked_text.replace('TARGETCIT', '<mask>')
        ground_truth_text = temp_masked_text.replace('TARGETCIT', temp_target_token)

        masked_cit_contexts_list.append(masked_with_mask_text)
        cit_contexts_list.append(ground_truth_text)
        masked_token_target_list.append(temp_target_token)

        if len(cit_contexts_list) == max_dataset_size:
            break

    count_masked_contexts_with_more_than_400_tokens(masked_cit_contexts_list)

    new_df_table = pd.DataFrame({'citation_context': cit_contexts_list, 'masked_cit_context': masked_cit_contexts_list,
                                 'masked_token_target': masked_token_target_list})
    new_df_table.to_csv(dataset_output_file)

    citations_for_vocab = list(set(masked_token_target_list))
    vocab_additions = pd.DataFrame({'additions_to_vocab': citations_for_vocab})
    vocab_additions.to_csv(vocab_output_file)

    print("--> Length of whole set: ", len(cit_contexts_list))
    print("--> Skip count: ", skip_count, "\n")
    print("--> Length of citations_for_vocab: ", len(citations_for_vocab), "\n")


def split_dataset():
    contexts_df = pd.read_csv(dataset_output_file)

    # Shuffle the DataFrame rows
    contexts_df = contexts_df.sample(frac=1)

    split_threshold = int(len(contexts_df) * 80 / 100)  # I have selected 20% as the eval set.

    # Split the df into train and eval sets
    df_train = contexts_df.iloc[:split_threshold, 1:]
    df_eval = contexts_df.iloc[split_threshold:, 1:]

    print("--> Length of train set: ", len(df_train))
    print("--> Length of eval set: ", len(df_eval), "\n")

    df_train.to_csv(train_set_output_file, index=False)
    df_eval.to_csv(eval_set_output_file, index=False)


tokenizer = RobertaTokenizer.from_pretrained("roberta-base", truncation=True, padding='max_length', max_length=500)


def count_masked_contexts_with_more_than_400_tokens(masked_cit_contexts):
    more_than_400_count = 0
    for m in masked_cit_contexts:
        tokenized_masked_text = tokenizer.encode(m)[1:-1]
        if len(tokenized_masked_text) > 400:
            more_than_400_count += 1
    print("--->> Number of masked contexts with more than 400 tokens =", more_than_400_count, "\n")

# **********************************************************************************************************


def create_300k_dataset_ignore_1s():
    contexts_df = pd.read_json(contexts_file)
    papers_df = pd.read_json(papers_file)

    cit_contexts_list = []
    masked_cit_contexts_list = []
    masked_token_target_list = []

    with open(appearance_count_per_cit_dict_file) as json_file:
        appearance_count_per_cit_dict = json.load(json_file)

    context_df_length = len(contexts_df.columns)
    remaining_indexes_without_1s = []
    for j in range(context_df_length):
        temp_context_row = contexts_df.iloc[:, j]
        temp_target_token = create_target_token_for_ref_paper_id(temp_context_row['refid'], papers_df)
        if temp_target_token == "":
            continue
        if appearance_count_per_cit_dict[temp_target_token] > 1:
            remaining_indexes_without_1s.append(j)

    print("\n-----> Length of remaining_indexes_without_1s =", len(remaining_indexes_without_1s), "\n")
    print("\n-----> Length of only 1s =", 3205210 - len(remaining_indexes_without_1s), "\n")

    chosen_300k_indexes = random.sample(remaining_indexes_without_1s, max_dataset_size)
    for i in chosen_300k_indexes:
        temp_context_row = contexts_df.iloc[:, i]

        temp_target_token = create_target_token_for_ref_paper_id(temp_context_row['refid'], papers_df)

        temp_masked_text = temp_context_row['masked_text']
        temp_masked_text = temp_masked_text.replace('OTHERCIT', '')

        masked_with_mask_text = temp_masked_text.replace('TARGETCIT', '<mask>')
        ground_truth_text = temp_masked_text.replace('TARGETCIT', temp_target_token)

        masked_cit_contexts_list.append(masked_with_mask_text)
        cit_contexts_list.append(ground_truth_text)
        masked_token_target_list.append(temp_target_token)

    count_masked_contexts_with_more_than_400_tokens(masked_cit_contexts_list)

    new_df_table = pd.DataFrame({'citation_context': cit_contexts_list, 'masked_cit_context': masked_cit_contexts_list,
                                 'masked_token_target': masked_token_target_list})
    new_df_table.to_csv(dataset_output_file)

    citations_for_vocab = list(set(masked_token_target_list))
    vocab_additions = pd.DataFrame({'additions_to_vocab': citations_for_vocab})
    vocab_additions.to_csv(vocab_output_file)

    print("--> Length of whole set: ", len(cit_contexts_list))
    # print("--> Skip count: ", skip_count, "\n")
    print("--> Length of citations_for_vocab: ", len(citations_for_vocab), "\n")


def create_300k_dataset_select_best_300k():
    contexts_df = pd.read_json(contexts_file)
    papers_df = pd.read_json(papers_file)

    cit_contexts_list = []
    masked_cit_contexts_list = []
    masked_token_target_list = []

    with open(appearance_count_per_cit_dict_file) as json_file:
        appearance_count_per_cit_dict = json.load(json_file)

    sorted_dict = dict(sorted(appearance_count_per_cit_dict.items(), key=lambda item: item[1], reverse=True))
    # print("-------> Sorted appearance_count_per_cit_dict:\n", sorted_dict, "\n\n")

    selected_citation_names = []
    temp_ref_count = 0
    for j in sorted_dict.keys():
        temp_ref_count += sorted_dict[j]
        selected_citation_names.append(j)
        if temp_ref_count >= max_dataset_size:
            break

    context_df_length = len(contexts_df.columns)
    for i in range(context_df_length):
        temp_context_row = contexts_df.iloc[:, i]

        temp_target_token = create_target_token_for_ref_paper_id(temp_context_row['refid'], papers_df)

        if temp_target_token not in selected_citation_names:
            continue

        temp_masked_text = temp_context_row['masked_text']
        temp_masked_text = temp_masked_text.replace('OTHERCIT', '')

        masked_with_mask_text = temp_masked_text.replace('TARGETCIT', '<mask>')
        ground_truth_text = temp_masked_text.replace('TARGETCIT', temp_target_token)

        masked_cit_contexts_list.append(masked_with_mask_text)
        cit_contexts_list.append(ground_truth_text)
        masked_token_target_list.append(temp_target_token)

        if len(cit_contexts_list) == max_dataset_size:
            break

    new_df_table = pd.DataFrame({'citation_context': cit_contexts_list, 'masked_cit_context': masked_cit_contexts_list,
                                 'masked_token_target': masked_token_target_list})
    new_df_table.to_csv(dataset_output_file)

    citations_for_vocab = list(set(masked_token_target_list))
    vocab_additions = pd.DataFrame({'additions_to_vocab': citations_for_vocab})
    vocab_additions.to_csv(vocab_output_file)

    print("--> Length of whole set: ", len(cit_contexts_list))
    print("--> Length of citations_for_vocab: ", len(citations_for_vocab), "\n")


if __name__ == '__main__':

    # create_300k_dataset_standard()
    # create_300k_dataset_ignore_1s()
    create_300k_dataset_select_best_300k()

    split_dataset()
