import pandas as pd
from dateutil.parser import parse
import re
from transformers import RobertaTokenizer

contexts_file = "../../../data_preprocessing/acl_200_original/contexts.json"
papers_file = "../../../data_preprocessing/acl_200_original/papers.json"

dataset_output_file = "acl200_new/context_dataset.csv"
vocab_output_file = "acl200_new/citation_item_list.csv"
train_set_output_file = "acl200_new/context_dataset_train.csv"
eval_set_output_file = "acl200_new/context_dataset_eval.csv"

context_limit = 100


def check_if_string_contains_year(marker):
    if len(marker) < 8:  # Eliminate smaller citation markers.
        return False
    match = re.match(r'.*([1-2][0-9]{3})', marker)
    if match is not None:
        return True
    return False


def create_target_token_for_ref_paper_id(marker_from_contexts_file):
    temp_marker = marker_from_contexts_file.replace('(', ' ').replace(')', ' ')

    if not check_if_string_contains_year(temp_marker):  # Skip marker without any years, e.g. "[S91]", "[Chodorov]"
        return ""
    year_from_marker_info = str(parse(temp_marker, fuzzy=True).year)

    authors_from_marker = marker_from_contexts_file.split(", ")[:-1]

    target_cit_token = ""
    if len(authors_from_marker) == 1:
        target_cit_token = authors_from_marker[0].split(" ")[-1] + ", " + year_from_marker_info  # No capitalize()
    elif len(authors_from_marker) == 2:
        target_cit_token = authors_from_marker[0].split(" ")[-1] + " and " + \
                           authors_from_marker[1].split(" ")[-1] + ", " + year_from_marker_info
    elif len(authors_from_marker) > 2:
        target_cit_token = authors_from_marker[0].split(" ")[-1] + " et al., " + year_from_marker_info

    return target_cit_token


def preprocess_dataset():
    contexts_df = pd.read_json(contexts_file)
    papers_df = pd.read_json(papers_file)

    masked_cit_contexts_list = []
    masked_token_target_list = []
    target_title_list = []
    target_abstract_list = []
    citing_title_list = []
    citing_abstract_list = []

    skip_count = 0
    for i in contexts_df:
        temp_context_row = contexts_df[i]

        marker_from_contexts_file = temp_context_row['marker']
        temp_target_token = create_target_token_for_ref_paper_id(marker_from_contexts_file)
        if temp_target_token == "":
            skip_count += 1
            continue

        temp_masked_context = temp_context_row['masked_text'].replace('TARGETCIT', '<mask>')

        trimmed_masked_context = trim_context_from_both_sides(temp_masked_context, context_length=context_limit)

        if trimmed_masked_context.find("<mask>") == -1:
            skip_count += 1
            continue

        ref_id = temp_context_row['context_id'].split("_")[1]

        target_title = papers_df[ref_id]["title"].replace("\n", "")
        target_abstract = papers_df[ref_id]["abstract"]
        target_abstract = shorten_abstract(target_abstract)

        target_title_list.append(target_title)
        target_abstract_list.append(target_abstract)

        citing_title = papers_df[temp_context_row['citing_id']]["title"].replace("\n", "")
        citing_abstract = papers_df[temp_context_row['citing_id']]["abstract"]
        citing_abstract = shorten_abstract(citing_abstract)

        citing_title_list.append(citing_title)
        citing_abstract_list.append(citing_abstract)

        masked_cit_contexts_list.append(trimmed_masked_context)
        masked_token_target_list.append(temp_target_token)

    new_df_table = pd.DataFrame({'masked_cit_context': masked_cit_contexts_list,
                                 'masked_token_target': masked_token_target_list,
                                 'citing_title': citing_title_list, 'citing_abstract': citing_abstract_list,
                                 'target_title': target_title_list, 'target_abstract': target_abstract_list})
    new_df_table.to_csv(dataset_output_file)

    citations_for_vocab = list(set(masked_token_target_list))
    vocab_additions = pd.DataFrame({'citation_items': citations_for_vocab})
    vocab_additions.to_csv(vocab_output_file)

    print("--> Length of whole set: ", len(masked_cit_contexts_list))
    print("--> Skip count: ", skip_count, "\n")
    print("--> Citation items count: ", len(citations_for_vocab), "\n")


def trim_context_from_both_sides(masked_context, context_length=100):
    tokenized_context = tokenizer.tokenize(masked_context)
    if len(tokenized_context) <= context_length:
        return masked_context

    mask_idx = tokenized_context.index("<mask>")
    half_context_len = int(context_length / 2)
    if mask_idx - half_context_len <= 0:
        shorter_context_tokenized = tokenized_context[:mask_idx + half_context_len]
    elif mask_idx + half_context_len >= len(tokenized_context):
        shorter_context_tokenized = tokenized_context[mask_idx - half_context_len:]
    else:
        shorter_context_tokenized = tokenized_context[mask_idx - half_context_len: mask_idx + half_context_len]

    shorter_context_masked = tokenizer.convert_tokens_to_string(shorter_context_tokenized)
    shorter_context_masked = shorter_context_masked.replace('<mask>', ' <mask> ')
    return shorter_context_masked


def shorten_abstract(temp_abstract, max_abstract_limit=200):
    tokenized_abstract = tokenizer.tokenize(temp_abstract)
    if len(tokenized_abstract) > max_abstract_limit:
        shortened_tokenized_abstract = tokenized_abstract[:max_abstract_limit]
        shortened_abstract = tokenizer.convert_tokens_to_string(shortened_tokenized_abstract)
        return shortened_abstract
    else:
        return temp_abstract


def split_dataset():
    contexts_df = pd.read_csv(dataset_output_file)

    # Shuffle the DataFrame rows
    contexts_df = contexts_df.sample(frac=1, random_state=42)

    split_threshold = int(len(contexts_df) * 80 / 100)  # I have selected 20% as the eval set.

    # Split the df into train and eval sets
    df_train = contexts_df.iloc[:split_threshold, 1:]
    df_eval = contexts_df.iloc[split_threshold:, 1:]

    print("--> Length of train set: ", len(df_train))
    print("--> Length of eval set: ", len(df_eval))

    df_train.to_csv(train_set_output_file, index=False)
    df_eval.to_csv(eval_set_output_file, index=False)


tokenizer = RobertaTokenizer.from_pretrained("roberta-base", truncation=True, padding='max_length', max_length=500)


if __name__ == '__main__':
    preprocess_dataset()

    split_dataset()
