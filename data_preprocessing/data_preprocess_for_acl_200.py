import pandas as pd
from dateutil.parser import parse
import re


contexts_file = "./acl_200_original/contexts.json"
papers_file = "./acl_200_original/papers.json"

dataset_output_file = "./acl_200/context_dataset.csv"
vocab_output_file = "./acl_200/additions_to_vocab.csv"
train_set_output_file = "./acl_200/context_dataset_train.csv"
eval_set_output_file = "./acl_200/context_dataset_eval.csv"


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


def place_mask_and_target_cit_on_ground_truth_context(ground_truth_context, target_cit_token,
                                                      marker_from_contexts_file):
    number_of_authors = len(marker_from_contexts_file.split(", ")[:-1])

    masked_context = ""
    unmasked_context = ""

    if number_of_authors > 2:
        split_target_cit = target_cit_token.split(" et al., ")
        regex_string = re.compile(rf"{split_target_cit[0]}(.{{0,10}}?){split_target_cit[1]}")

        masked_context = re.sub(regex_string, ' <mask> ', ground_truth_context)
        unmasked_context = re.sub(regex_string, f' {target_cit_token} ', ground_truth_context)
        """print(ground_truth_context, "\n", masked_context, "\n--------\n")
        if masked_context.find("<mask>") == -1:
            print("===========>>>>>> FAILED TO FIND MASK! ---->   ", target_cit_token)"""

    elif number_of_authors == 2:
        temp_split_target_cit = target_cit_token.split(" and ")
        temp_split_target_cit_2 = temp_split_target_cit[1].split(", ")
        split_target_cit = [temp_split_target_cit[0], temp_split_target_cit_2[0], temp_split_target_cit_2[1]]
        regex_string = re.compile(rf"{split_target_cit[0]}(.{{0,10}}?){split_target_cit[1]}(.{{0,10}}?)"
                                  rf"{split_target_cit[2]}")

        masked_context = re.sub(regex_string, ' <mask> ', ground_truth_context)
        unmasked_context = re.sub(regex_string, f' {target_cit_token} ', ground_truth_context)

    elif number_of_authors == 1:
        split_target_cit = target_cit_token.split(", ")
        regex_string = re.compile(rf"{split_target_cit[0]}(.{{0,10}}?){split_target_cit[1]}")

        masked_context = re.sub(regex_string, ' <mask> ', ground_truth_context)
        unmasked_context = re.sub(regex_string, f' {target_cit_token} ', ground_truth_context)

    if masked_context.find("<mask>") == -1:
        masked_context = ""
        unmasked_context = ""

    return masked_context, unmasked_context


def preprocess_dataset():
    contexts_df = pd.read_json(contexts_file)

    cit_contexts_list = []
    masked_cit_contexts_list = []
    masked_token_target_list = []

    skip_count = 0
    for i in contexts_df:
        temp_context_row = contexts_df[i]
        ground_truth_context = temp_context_row['citation_context']

        marker_from_contexts_file = temp_context_row['marker']
        temp_target_token = create_target_token_for_ref_paper_id(marker_from_contexts_file)
        if temp_target_token == "":
            skip_count += 1
            continue

        temp_masked_text, temp_unmasked_text = \
            place_mask_and_target_cit_on_ground_truth_context(ground_truth_context, temp_target_token,
                                                              marker_from_contexts_file)
        if temp_masked_text == "" or temp_unmasked_text == "":
            skip_count += 1
            continue

        masked_token_target_list.append(temp_target_token)
        masked_cit_contexts_list.append(temp_masked_text)
        cit_contexts_list.append(temp_unmasked_text)

    new_df_table = pd.DataFrame({'citation_context': cit_contexts_list, 'masked_cit_context': masked_cit_contexts_list,
                                 'masked_token_target': masked_token_target_list})
    new_df_table.to_csv(dataset_output_file)

    citations_for_vocab = list(set(masked_token_target_list))
    vocab_additions = pd.DataFrame({'additions_to_vocab': citations_for_vocab})
    vocab_additions.to_csv(vocab_output_file)

    print("--> Length of whole set: ", len(cit_contexts_list))
    print("--> Skip count: ", skip_count, "\n")
    print("--> Additional vocab size: ", len(citations_for_vocab), "\n")


def split_dataset():
    contexts_df = pd.read_csv(dataset_output_file)

    # Shuffle the DataFrame rows
    contexts_df = contexts_df.sample(frac=1)

    split_threshold = int(len(contexts_df) * 80 / 100)  # I have selected 20% as the eval set.

    # Split the df into train and eval sets
    df_train = contexts_df.iloc[:split_threshold, 1:]
    df_eval = contexts_df.iloc[split_threshold:, 1:]

    print("--> Length of train set: ", len(df_train))
    print("--> Length of eval set: ", len(df_eval))

    df_train.to_csv(train_set_output_file, index=False)
    df_eval.to_csv(eval_set_output_file, index=False)


if __name__ == '__main__':
    preprocess_dataset()

    split_dataset()
