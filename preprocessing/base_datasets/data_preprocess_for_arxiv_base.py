import pandas as pd
import random
from transformers import RobertaTokenizer


contexts_file = "../original_datasets/arxiv_original/contexts.json"
papers_file = "../original_datasets/arxiv_original/papers.json"

dataset_output_file = "./arxiv_base/context_dataset.csv"
vocab_output_file = "./arxiv_base/citation_item_list.csv"
train_set_output_file = "./arxiv_base/context_dataset_train.csv"
eval_set_output_file = "./arxiv_base/context_dataset_eval.csv"
random.seed(42)

max_token_limit = 300


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

        ground_truth_text, masked_with_mask_text = shorten_unmasked_context_with_more_than_k_tokens(
            ground_truth_text, masked_with_mask_text, k=max_token_limit)

        if ground_truth_text == "X" or masked_with_mask_text == "X":
            skip_count += 1
            continue

        masked_cit_contexts_list.append(masked_with_mask_text)
        cit_contexts_list.append(ground_truth_text)
        masked_token_target_list.append(temp_target_token)

    count_masked_contexts_with_more_than_k_tokens(masked_cit_contexts_list, k=max_token_limit)

    new_df_table = pd.DataFrame({'citation_context': cit_contexts_list, 'masked_cit_context': masked_cit_contexts_list,
                                 'masked_token_target': masked_token_target_list})
    new_df_table.to_csv(dataset_output_file)

    citations_for_vocab = list(set(masked_token_target_list))
    vocab_additions = pd.DataFrame({'citation_items': citations_for_vocab})
    vocab_additions.to_csv(vocab_output_file)

    print("--> Length of whole set: ", len(cit_contexts_list))
    print("--> Skip count: ", skip_count, "\n")
    print("--> Citation item size: ", len(citations_for_vocab), "\n")


def split_dataset():
    contexts_df = pd.read_csv(dataset_output_file)

    # Shuffle the DataFrame rows
    contexts_df = contexts_df.sample(frac=1, random_state=42)

    split_threshold = int(len(contexts_df) * 80 / 100)  # I have selected 20% as the eval set.

    # Split the df into train and eval sets
    df_train = contexts_df.iloc[:split_threshold, 1:]
    df_eval = contexts_df.iloc[split_threshold:, 1:]

    print("--> Length of train set: ", len(df_train))
    print("--> Length of eval set: ", len(df_eval), "\n")

    df_train.to_csv(train_set_output_file, index=False)
    df_eval.to_csv(eval_set_output_file, index=False)


tokenizer = RobertaTokenizer.from_pretrained("roberta-base", truncation=True, padding='max_length', max_length=500)


def count_masked_contexts_with_more_than_k_tokens(masked_cit_contexts, k=400):
    more_than_k_count = 0
    for m in masked_cit_contexts:
        tokenized_masked_text = tokenizer.encode(m)[1:-1]
        if len(tokenized_masked_text) > k:
            more_than_k_count += 1
    print(f"--->> Number of masked contexts with more than {k} tokens =", more_than_k_count, "\n")


def shorten_unmasked_context_with_more_than_k_tokens(unmasked_cit_context, masked_cit_context, k=400):
    tokenized_unmasked_text = tokenizer.tokenize(unmasked_cit_context)
    if len(tokenized_unmasked_text) > k:
        diff_from_k = len(tokenized_unmasked_text) - k
        if diff_from_k > 150:  # Eliminate examples that need to be cut too much
            return "X", "X"
        cut_amount = int((diff_from_k + 3) / 2)  # Make the cut amount slightly larger thanks to +3.
        shortened_tokenized_unmasked = tokenized_unmasked_text[cut_amount:-cut_amount]
        shortened_unmasked_str = tokenizer.convert_tokens_to_string(shortened_tokenized_unmasked)

        temp_tokenized_masked = tokenizer.tokenize(masked_cit_context)
        shortened_tokenized_masked = temp_tokenized_masked[cut_amount:-cut_amount]
        shortened_masked_str = tokenizer.convert_tokens_to_string(shortened_tokenized_masked)
        return shortened_unmasked_str, shortened_masked_str
    return unmasked_cit_context, masked_cit_context


if __name__ == '__main__':
    preprocess_dataset()

    split_dataset()
