import pandas as pd
import re
import random
from transformers import RobertaTokenizer


contexts_file = "./refseer_original/contexts.json"
papers_file = "./refseer_original/papers.json"

dataset_output_file = "./refseer200k/context_dataset.csv"
vocab_output_file = "./refseer200k/additions_to_vocab.csv"
train_set_output_file = "./refseer200k/context_dataset_train.csv"
eval_set_output_file = "./refseer200k/context_dataset_eval.csv"

max_dataset_size = 200000
random.seed(42)
max_token_limit = 200


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


# ref_id keys here are actually citing_ids from the contexts of refseer. Their type should be integers.
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

        # For refseer; I have to use 'citing_id' values instead of 'refid' values unlike peerread. --> I WAS WRONG!!!
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

        masked_raw_text = re.sub(r'=-=(.*?)-=-', ' <mask> ', temp_raw_text)
        ground_truth_text = re.sub(r'=-=(.*?)-=-', f' {temp_target_token} ', temp_raw_text)

        ground_truth_text, masked_raw_text = shorten_unmasked_context_with_more_than_k_tokens(
            ground_truth_text, masked_raw_text, k=max_token_limit)

        masked_cit_contexts_list.append(masked_raw_text)
        cit_contexts_list.append(ground_truth_text)
        masked_token_target_list.append(temp_target_token)

        if len(cit_contexts_list) == max_dataset_size:
            break

    count_masked_contexts_with_more_than_k_tokens(masked_cit_contexts_list, k=max_token_limit)

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


def count_masked_contexts_with_more_than_k_tokens(masked_cit_contexts, k=300):
    more_than_400_count = 0
    for m in masked_cit_contexts:
        tokenized_masked_text = tokenizer.encode(m)[1:-1]
        if len(tokenized_masked_text) > k:
            more_than_400_count += 1
    print(f"--->> Number of masked contexts with more than {k} tokens = {more_than_400_count}\n")


def shorten_unmasked_context_with_more_than_k_tokens(unmasked_cit_context, masked_cit_context, k=300):
    tokenized_unmasked_text = tokenizer.tokenize(unmasked_cit_context)
    if len(tokenized_unmasked_text) > k:
        diff_from_k = len(tokenized_unmasked_text) - k
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
