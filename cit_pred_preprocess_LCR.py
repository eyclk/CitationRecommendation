import pandas as pd


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
    contexts_json = pd.read_json("./contexts.json")
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
    new_df_table.to_csv("./context_only_dataset.csv")

    citations_for_vocab = list(set(citations_for_vocab))
    vocab_additions = pd.DataFrame({'additions_to_vocab': citations_for_vocab})
    vocab_additions.to_csv("./additions_to_vocab.csv")


def split_dataset():
    contexts_df = pd.read_csv("./context_only_dataset.csv")

    # Shuffle the DataFrame rows
    contexts_df = contexts_df.sample(frac=1)

    split_threshold = int(len(contexts_df) * 80 / 100)  # I have selected 20% as the eval set.

    # Split the df into train and eval sets
    df_train = contexts_df.iloc[:split_threshold, 1:]
    df_eval = contexts_df.iloc[split_threshold:, 1:]

    print(len(df_train))
    print(len(df_eval))

    df_train.to_csv("./context_dataset_train.csv", index=False)
    df_eval.to_csv("./context_dataset_eval.csv", index=False)


if __name__ == '__main__':
    # paper_json = pd.read_json("cit_data/papers.json")
    # print(paper_json["N13-1016"], "\n\n")

    # preprocess_contexts_json()

    # preprocess_dataset_for_masking()

    split_dataset()

    """context_json = pd.read_json("cit_data/contexts.json")
    print(context_json["S14-1013_P08-1028_0"], "\n\n")

    print(context_json["S14-1013_P08-1028_0"]["masked_text"], "\n\n")
    print(context_json["S14-1013_P08-1028_0"]["citation_context"], "\n\n")"""
