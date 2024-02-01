import pandas as pd
import re
import random
from transformers import RobertaTokenizer


# ====> Global info refers to title and abstract.

contexts_file = "../data_preprocessing/refseer_original/contexts.json"
papers_file = "../data_preprocessing/refseer_original/papers.json"

dataset_output_file = "./refseer_global/context_dataset.csv"
vocab_output_file = "./refseer_global/additions_to_vocab.csv"
train_set_output_file = "./refseer_global/context_dataset_train.csv"
eval_set_output_file = "./refseer_global/context_dataset_eval.csv"

random.seed(42)

context_len = 50  # 200 version is not necessary for this dataset.


# This check exists to check raw data just in case. However, all raw data already contains =-=, -=-.
def check_if_raw_text_has_special_tags(raw_text):
    if raw_text.find('=-=') == -1:
        return False
    if raw_text.find('-=-') == -1:
        return False
    return True


def assign_appropriate_year_for_null_years(ref_id, author_names):
    if ref_id not in dict_missing_years_for_refid.keys():
        random_year = random.randint(1960, 2014)
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


# ref_id keys here are still for the masked citation tokens, and they show what these tokens refer to.
# Their type should be integers. I WAS PREVIOUSLY WRONG.
def create_target_token_for_ref_paper_id(ref_id, papers_df):
    target_cit_token = ""
    temp_paper_info_row = papers_df[int(ref_id)]
    authors_from_paper_info = temp_paper_info_row['authors']

    if temp_paper_info_row['year'] == 'NULL':
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


# IMPORTANT: Default max_token_limit have been reduced to 400!!!
def concatenate_title_and_abstract_while_making_context_shorter(masked_context, temp_target, ref_id,
                                                                papers_df, context_length=50, max_token_limit=400):
    if int(ref_id) not in papers_df.columns:
        return "", ""

    temp_title = papers_df.at["title", int(ref_id)]
    temp_abstract = papers_df.at["abstract", int(ref_id)]

    tokenized_context = tokenizer.tokenize(masked_context)
    mask_idx = tokenized_context.index("<mask>")
    half_context_len = int(context_length / 2)
    shorter_context_tokenized = tokenized_context[mask_idx - half_context_len: mask_idx + half_context_len]

    shorter_context_masked = tokenizer.convert_tokens_to_string(shorter_context_tokenized)
    # shorter_context_masked = re.sub('<mask>', ' <mask>', shorter_context_masked)
    shorter_context_masked = shorter_context_masked.replace('<mask>', ' <mask>')

    left_context = tokenized_context[mask_idx - half_context_len: mask_idx]
    right_context = tokenized_context[mask_idx + 1: mask_idx + half_context_len]
    target_tokenized = tokenizer.tokenize(temp_target)
    unmasked_tokenized = left_context + target_tokenized + right_context

    shorter_context_unmasked = tokenizer.convert_tokens_to_string(unmasked_tokenized)
    # shorter_context_unmasked = re.sub('<mask>', ' <mask>', shorter_context_unmasked)
    shorter_context_unmasked = shorter_context_unmasked.replace(temp_target, ' '+temp_target)

    masked_context_with_global_info = shorter_context_masked + " </s> " + temp_title + " </s> " + temp_abstract
    tokenized_with_global_info_masked = tokenizer.tokenize(masked_context_with_global_info)
    if len(tokenized_with_global_info_masked) > max_token_limit:
        trimmed_tokenized_with_global_info_masked = tokenized_with_global_info_masked[:max_token_limit]
        masked_context_with_global_info = tokenizer.convert_tokens_to_string(trimmed_tokenized_with_global_info_masked)

    unmasked_context_with_global_info = shorter_context_unmasked + " </s> " + temp_title + " </s> " + temp_abstract
    tokenized_with_global_info_unmasked = tokenizer.tokenize(unmasked_context_with_global_info)
    if len(tokenized_with_global_info_unmasked) > max_token_limit:
        trimmed_tokenized_with_global_info_unmasked = tokenized_with_global_info_unmasked[:max_token_limit]
        unmasked_context_with_global_info = tokenizer.convert_tokens_to_string(
            trimmed_tokenized_with_global_info_unmasked)

    return masked_context_with_global_info, unmasked_context_with_global_info


def preprocess_dataset():
    contexts_df = pd.read_json(contexts_file)
    papers_df = pd.read_json(papers_file)

    cit_contexts_list = []
    masked_cit_contexts_list = []
    masked_token_target_list = []

    skip_count = 0
    context_df_length = len(contexts_df.columns)
    for i in range(context_df_length):
        temp_context_row = contexts_df.iloc[:, i]

        # For refseer; I have to use 'citing_id' values instead of 'refid' values unlike peerread!!!
        # THIS WAS WRONG! I still need to use refid for citation token's values (similar to other datasets).
        temp_target_token = create_target_token_for_ref_paper_id(temp_context_row['refid'], papers_df)
        if temp_target_token == "":
            skip_count += 1
            continue

        temp_raw_text = temp_context_row['raw']
        if not check_if_raw_text_has_special_tags(temp_raw_text):  # This if branch is never entered.
            skip_count += 1
            continue

        # Some examples in the dataset contain '\\' substrings that cause problems with re package. They get replaced.
        temp_target_token = temp_target_token.replace("\\", "//")

        temp_masked_text = re.sub(r'=-=(.*?)-=-', ' <mask> ', temp_raw_text)
        # ground_truth_text = re.sub(r'=-=(.*?)-=-', f' {temp_target_token} ', temp_raw_text)

        masked_text_global, ground_truth_text_global = concatenate_title_and_abstract_while_making_context_shorter(
            temp_masked_text, temp_target_token, temp_context_row['refid'], papers_df, context_length=context_len)

        if masked_text_global == "" or ground_truth_text_global == "":
            skip_count += 1
            continue

        masked_cit_contexts_list.append(masked_text_global)
        cit_contexts_list.append(ground_truth_text_global)
        masked_token_target_list.append(temp_target_token)

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


if __name__ == '__main__':
    preprocess_dataset()

    split_dataset()
