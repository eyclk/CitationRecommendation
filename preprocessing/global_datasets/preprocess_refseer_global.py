import pandas as pd
import re
import random
from transformers import RobertaTokenizer
from tqdm import tqdm


contexts_file = "../original_datasets/refseer_original/contexts.json"
papers_file = "../original_datasets/refseer_original/papers.json"

dataset_output_file = "./refseer_global/context_dataset.csv"
vocab_output_file = "./refseer_global/citation_item_list.csv"
train_set_output_file = "./refseer_global/context_dataset_train.csv"
eval_set_output_file = "./refseer_global/context_dataset_eval.csv"

random.seed(42)

context_limit = 100


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


# ref_id keys here are for the masked citation tokens, and they show what these tokens refer to.
# Their type should be integers.
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

    masked_cit_contexts_list = []
    masked_token_target_list = []
    target_title_list = []
    target_abstract_list = []
    citing_title_list = []
    citing_abstract_list = []

    skip_count = 0
    context_df_length = len(contexts_df.columns)

    # total_count = -1  # DELETE

    for i in tqdm(range(context_df_length)):

        """total_count += 1  # DELETE
        if total_count == 500:  # DELETE
            break  # DELETE"""

        temp_context_row = contexts_df.iloc[:, i]

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

        temp_masked_context = re.sub(r'=-=(.*?)-=-', ' <mask> ', temp_raw_text)
        trimmed_masked_context = trim_context_from_both_sides(temp_masked_context, context_length=context_limit)

        ref_id = temp_context_row['refid']
        citing_id = temp_context_row['citing_id']

        if int(ref_id) not in papers_df.columns or int(citing_id) not in papers_df.columns:
            skip_count += 1
            continue

        target_title = papers_df.at["title", int(ref_id)]
        target_abstract = papers_df.at["abstract", int(ref_id)]
        target_abstract = shorten_abstract(target_abstract)

        target_title_list.append(target_title)
        target_abstract_list.append(target_abstract)

        citing_title = papers_df.at["title", int(citing_id)]
        citing_abstract = papers_df.at["abstract", int(citing_id)]
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
    citations = pd.DataFrame({'additions_to_vocab': citation_item_list})
    citations.to_csv(vocab_output_file)

    print("--> Length of whole set: ", len(masked_cit_contexts_list))
    print("--> Skip count: ", skip_count, "\n")
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
    print("--> Length of eval set: ", len(df_eval), "\n")

    df_train.to_csv(train_set_output_file, index=False)
    df_eval.to_csv(eval_set_output_file, index=False)


tokenizer = RobertaTokenizer.from_pretrained("roberta-base", truncation=True, padding='max_length', max_length=500)


if __name__ == '__main__':
    """contexts_df = pd.read_json(contexts_file)
    print(contexts_df.head(2), "\n")
    context_df_length = len(contexts_df.columns)
    for i in range(context_df_length):
        temp_context_row = contexts_df.iloc[:, i]
        print(temp_context_row)
        break

    papers_df = pd.read_json(papers_file)
    print("\n", papers_df.head(2), "\n")
    for i in range(len(papers_df.columns)):
        temp_context_row = papers_df.iloc[:, i]
        print(temp_context_row)
        break"""

    preprocess_dataset()

    split_dataset()
