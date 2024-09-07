import pandas as pd
from transformers import RobertaTokenizer


contexts_file = "../original_datasets/peerread_original/contexts.json"
papers_file = "../original_datasets/peerread_original/papers.json"

dataset_output_file = "peerread_global/context_dataset.csv"
vocab_output_file = "peerread_global/citation_item_list.csv"
train_set_output_file = "peerread_global/context_dataset_train.csv"
eval_set_output_file = "peerread_global/context_dataset_eval.csv"

context_limit = 100


def create_target_token_for_ref_paper_id(ref_id, papers_df):
    temp_paper_info_row = papers_df[ref_id]
    year_from_paper_info = str(int(float(temp_paper_info_row['year'])))
    authors_from_paper_info = temp_paper_info_row['authors']

    target_cit_token = ""
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

    masked_cit_contexts_list = []
    masked_token_target_list = []
    target_title_list = []
    target_abstract_list = []
    citing_title_list = []
    citing_abstract_list = []

    context_df_length = len(contexts_df.columns)
    for i in range(context_df_length):
        temp_context_row = contexts_df.iloc[:, i]

        temp_masked_context = temp_context_row['masked_text'].replace('TARGETCIT', ' <mask> ')
        trimmed_masked_context = trim_context_from_both_sides(temp_masked_context, context_length=context_limit)

        temp_target_token = create_target_token_for_ref_paper_id(temp_context_row['refid'], papers_df)

        target_title = papers_df[temp_context_row['refid']]["title"]
        target_abstract = papers_df[temp_context_row['refid']]["abstract"]
        target_abstract = shorten_abstract(target_abstract)

        target_title_list.append(target_title)
        target_abstract_list.append(target_abstract)

        citing_title = papers_df[temp_context_row['citing_id']]["title"]
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

    citation_item_list = list(set(masked_token_target_list))
    citations = pd.DataFrame({'citation_items': citation_item_list})
    citations.to_csv(vocab_output_file)

    print("--> Length of whole set: ", len(masked_cit_contexts_list))
    print("--> Citation item size: ", len(citation_item_list), "\n")


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
