import pandas as pd


contexts_file = "./peerread_original/contexts.json"
papers_file = "./peerread_original/papers.json"

dataset_output_file = "./peerread/context_dataset.csv"
vocab_output_file = "./peerread/additions_to_vocab.csv"
train_set_output_file = "./peerread/context_dataset_train.csv"
eval_set_output_file = "./peerread/context_dataset_eval.csv"


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
    
    cit_contexts_list = []
    masked_cit_contexts_list = []
    masked_token_target_list = []

    context_df_length = len(contexts_df.columns)
    for i in range(context_df_length):
        temp_context_row = contexts_df.iloc[:, i]

        temp_masked_text = temp_context_row['masked_text'].replace('TARGETCIT', '<mask>')
        masked_cit_contexts_list.append(temp_masked_text)

        temp_target_token = create_target_token_for_ref_paper_id(temp_context_row['refid'], papers_df)
        masked_token_target_list.append(temp_target_token)

        temp_unmasked_text = temp_context_row['masked_text'].replace('TARGETCIT', temp_target_token)
        cit_contexts_list.append(temp_unmasked_text)

    new_df_table = pd.DataFrame({'citation_context': cit_contexts_list, 'masked_cit_context': masked_cit_contexts_list,
                                 'masked_token_target': masked_token_target_list})
    new_df_table.to_csv(dataset_output_file)

    citations_for_vocab = list(set(masked_token_target_list))
    vocab_additions = pd.DataFrame({'additions_to_vocab': citations_for_vocab})
    vocab_additions.to_csv(vocab_output_file)


if __name__ == '__main__':
    preprocess_dataset()

    split_dataset()
