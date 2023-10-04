import pandas as pd
# from dateutil.parser import parse
# import re


contexts_file = "./acl_200_original/contexts.json"
papers_file = "./acl_200_original/papers.json"

dataset_output_file = "./acl_200_alt/context_dataset.csv"
vocab_output_file = "./acl_200_alt/additions_to_vocab.csv"
train_set_output_file = "./acl_200_alt/context_dataset_train.csv"
eval_set_output_file = "./acl_200_alt/context_dataset_eval.csv"


def replace_target_and_other_cits(cit_context, masked_token_marker):
    skip_flag = False

    marker_idx = cit_context.find(masked_token_marker)
    if len(masked_token_marker) < 8:
        # Eliminate citation tokens with 6 letters or fewer because they generally look like this: "[42]"
        skip_flag = True
        return "", "", skip_flag

    if masked_token_marker != cit_context[marker_idx: marker_idx+len(masked_token_marker)]:
        new_marker = masked_token_marker
        if masked_token_marker.count(',') >= 3:
            split_marker = masked_token_marker.split(',')
            new_marker = split_marker[0] + ' et al.,' + split_marker[-1]
        elif masked_token_marker.count(',') == 2:
            new_marker = masked_token_marker.replace(', ', ' and ', 1)

        marker_idx = cit_context.find(new_marker)
        if new_marker != cit_context[marker_idx: marker_idx + len(new_marker)]:
            if len(new_marker.split(', ')) == 2:
                split_marker = new_marker.split(', ')
                new_marker = split_marker[0] + ' (' + split_marker[1]

            marker_idx = cit_context.find(new_marker)
            if new_marker != cit_context[marker_idx: marker_idx + len(new_marker)]:
                new_marker = new_marker.replace(' (', ' ')

                marker_idx = cit_context.find(new_marker)
                if new_marker != cit_context[marker_idx: marker_idx + len(new_marker)]:
                    if 'and' in new_marker:
                        new_marker = new_marker.replace('and', '&')

                    marker_idx = cit_context.find(new_marker)
                    if new_marker != cit_context[marker_idx: marker_idx + len(new_marker)]:
                        # print("new_marker -->", new_marker)
                        # print("***** Could not find masked token marker inside the given true citation context!!!")
                        # print("---> ", cit_context, "\n\n")
                        skip_flag = True

        masked_token_marker = new_marker

    new_masked_context = cit_context.replace(masked_token_marker, "<mask>")
    return new_masked_context, masked_token_marker, skip_flag


def check_if_mask_token_is_near_the_end(masked_context):
    if len(masked_context) == 0:
        return True
    mask_token_idx = masked_context.find("<mask>")
    context_length = len(masked_context)
    if mask_token_idx / context_length > 0.85:
        return True
    return False


def preprocess_dataset_for_masking():
    contexts_json = pd.read_json(contexts_file)
    cit_context_list = []
    new_masked_context_list = []

    citations_for_vocab = []

    # loop_count = 0
    skip_count = 0
    for i in contexts_json:
        # print("--> Len of citation_contexts: ", len(contexts_json[i]["citation_context"]), "\n")
        temp_context = contexts_json[i]["citation_context"]
        new_masked_context, masked_token_marker, skip_flag = replace_target_and_other_cits(temp_context,
                                                                                           contexts_json[i]["marker"])
        if check_if_mask_token_is_near_the_end(new_masked_context):
            skip_count += 1
            continue

        if not skip_flag:
            new_masked_context_list.append(new_masked_context)
            cit_context_list.append(temp_context)
            if '(' in masked_token_marker:
                citations_for_vocab.append(masked_token_marker + ')')
            else:
                citations_for_vocab.append(masked_token_marker)
        else:
            skip_count += 1

        """loop_count += 1
        if loop_count > 500:
            break"""

    print("====> Skip count = ", skip_count, "\n")
    print("====> Length of new_masked_context_list = ", len(new_masked_context_list), "\n")

    new_df_table = pd.DataFrame({'citation_context': cit_context_list, 'masked_cit_context': new_masked_context_list,
                                 'masked_token_target': citations_for_vocab})
    new_df_table.to_csv(dataset_output_file)

    citations_for_vocab = list(set(citations_for_vocab))
    vocab_additions = pd.DataFrame({'additions_to_vocab': citations_for_vocab})
    vocab_additions.to_csv(vocab_output_file)


"""def check_if_string_contains_year(marker):
    if len(marker) < 8:  # Eliminate smaller citation markers.
        return False
    match = re.match(r'.*([1-2][0-9]{3})', marker)
    if match is not None:
        return True
    return False


def create_target_token_for_ref_paper_id(ref_id, papers_df, marker_from_contexts_file):
    temp_paper_info_row = papers_df[ref_id]
    authors_from_paper_info = temp_paper_info_row['authors']
    temp_marker = marker_from_contexts_file.replace('(', ' ').replace(')', ' ')

    if not check_if_string_contains_year(temp_marker):  # Skip marker without any years, e.g. "[S91]", "[Chodorov]"
        return ""
    year_from_marker_info = str(parse(temp_marker, fuzzy=True).year)

    target_cit_token = ""
    if len(authors_from_paper_info) == 1:
        target_cit_token = authors_from_paper_info[0].split(" ")[-1].capitalize() + ", " + year_from_marker_info
    elif len(authors_from_paper_info) == 2:
        target_cit_token = authors_from_paper_info[0].split(" ")[-1].capitalize() + " and " + \
                           authors_from_paper_info[1].split(" ")[-1].capitalize() + ", " + year_from_marker_info
    elif len(authors_from_paper_info) > 2:
        target_cit_token = authors_from_paper_info[0].split(" ")[-1].capitalize() + " et al., " + year_from_marker_info

    return target_cit_token


def preprocess_dataset():
    contexts_df = pd.read_json(contexts_file)
    papers_df = pd.read_json(papers_file)

    cit_contexts_list = []
    masked_cit_contexts_list = []
    masked_token_target_list = []

    skip_count = 0
    for i in contexts_df:
        temp_context_row = contexts_df[i]

        temp_masked_text = temp_context_row['masked_text'].replace('TARGETCIT', '<mask>')

        marker_from_contexts_file = temp_context_row['marker']
        temp_target_token = create_target_token_for_ref_paper_id(temp_context_row['refid'], papers_df,
                                                                 marker_from_contexts_file)
        if temp_target_token == "":
            skip_count += 1
            continue
        masked_token_target_list.append(temp_target_token)
        masked_cit_contexts_list.append(temp_masked_text)

        temp_unmasked_text = temp_context_row['masked_text'].replace('TARGETCIT', temp_target_token)
        cit_contexts_list.append(temp_unmasked_text)

    new_df_table = pd.DataFrame({'citation_context': cit_contexts_list, 'masked_cit_context': masked_cit_contexts_list,
                                 'masked_token_target': masked_token_target_list})
    new_df_table.to_csv(dataset_output_file)

    citations_for_vocab = list(set(masked_token_target_list))
    vocab_additions = pd.DataFrame({'additions_to_vocab': citations_for_vocab})
    vocab_additions.to_csv(vocab_output_file)

    print("--> Length of whole set: ", len(cit_contexts_list))
    print("--> Skip count: ", skip_count)"""


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
    # paper_json = pd.read_json("cit_data/papers.json")
    # print(paper_json["N13-1016"], "\n\n")

    """context_json = pd.read_json("refseer_original/contexts.json")
    print(context_json.iloc[0], "\n\n")

    context_json = pd.read_json("peerread_original/contexts.json")
    print(context_json.iloc[0], "\n\n")
    print(context_json["1606.03622v1_1409.3215v1_0"], "\n\n")
    context_json = pd.read_json("acl_200_original/contexts.json")
    print(context_json.iloc[0], "\n\n")"""
    # ---------------------------

    preprocess_dataset_for_masking()

    split_dataset()
