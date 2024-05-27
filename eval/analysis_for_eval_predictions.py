from transformers import RobertaForMaskedLM, RobertaTokenizer, pipeline
from datasets import Dataset
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--trained_model_path", type=str, help="Path of the local trained model")
parser.add_argument("--dataset_path", type=str, default="", help="Path to the folder of the dataset")
parser.add_argument("--eval_path", type=str, help="Path to the evaluation set of the dataset")
parser.add_argument("--max_token_limit", type=int, default=400, help="Max amount allowed for tokens used evaluation")

parser.add_argument("--first_analysed_example_idx", type=int, default=50, help="The index of the first example "
                                                                               "context that will be taken from the "
                                                                               "eval set")
parser.add_argument("--last_analysed_example_idx", type=int, default=55, help="The index of the last example context "
                                                                              "that will be taken from the eval set")
parser.add_argument("--top_k_number", type=int, default=10, help="The number of predictions to generate per context")
parser.add_argument("--eval_set_max_mapping_count", type=int, default=500, help="The number of rows to read and map "
                                                                                "from the eval set. If all rows are "
                                                                                "mapped running time increases too "
                                                                                "much.")


def tokenizer_function(tknizer, inp_data, col_name):
    return tknizer(inp_data[col_name], truncation=True, padding='max_length', max_length=max_token_limit)


def read_eval_dataset(tknizer):
    cit_df = pd.read_csv(eval_set_path)
    input_texts = []
    label_texts = []
    masked_targets = []

    row_count = -1
    for _, i in cit_df.iterrows():
        row_count += 1
        if row_count > eval_set_max_mapping_count:
            break
        input_texts.append(i['masked_cit_context'])
        label_texts.append(i['citation_context'])
        masked_targets.append(i['masked_token_target'])

    df_text_list = pd.DataFrame(input_texts, columns=['input_ids'])
    data_input_ids = Dataset.from_pandas(df_text_list)
    tokenized_input_ids = data_input_ids.map(
        lambda batch: tokenizer_function(tknizer, batch, 'input_ids'), batched=True)

    df_label_list = pd.DataFrame(label_texts, columns=['labels'])
    data_labels = Dataset.from_pandas(df_label_list)
    tokenized_labels = data_labels.map(
        lambda batch: tokenizer_function(tknizer, batch, 'labels'), batched=True)

    tokenized_data = tokenized_input_ids.add_column('labels', tokenized_labels['input_ids'])

    raw_and_tokenized_data = tokenized_data.add_column('masked_cit_context', input_texts)
    raw_and_tokenized_data = raw_and_tokenized_data.add_column('citation_context', label_texts)
    raw_and_tokenized_data = raw_and_tokenized_data.add_column('masked_token_target', masked_targets)

    return raw_and_tokenized_data


def shorten_masked_context_for_limit_if_necessary(masked_text):
    tokenized_text = tokenizer.tokenize(masked_text)
    if len(tokenized_text) > max_token_limit:  # Shorten texts with more than the max limit.
        exceeding_char_count = 5  # Always start with 5 extra characters just in case.
        for i in range(max_token_limit - 1, len(tokenized_text)):
            exceeding_char_count += len(tokenized_text[i])
        shortened_text = masked_text[:-exceeding_char_count]
        return shortened_text
    return masked_text


def display_top_k_predictions():
    for j in range(len(all_preds)):
        print(f"\n------------------------------------------------\n************* Example Context {j} *************")
        print(f"----->>> Masked context: {input_texts_for_test[j]}\n"
              f"----->>> Ground truth value: {masked_token_targets[j]}\n")
        temp_preds = all_preds[j]

        for p in range(len(temp_preds)):
            print(f"Pred {p} for Example Context {j}:")
            print(temp_preds[p], "\n\n")


if __name__ == '__main__':
    args = parser.parse_args()

    local_model_path = args.trained_model_path
    max_token_limit = args.max_token_limit

    first_eval_idx = args.first_analysed_example_idx
    last_eval_idx = args.last_analysed_example_idx
    top_k = args.top_k_number
    eval_set_max_mapping_count = args.eval_set_max_mapping_count

    dataset_folder = args.dataset_path
    if dataset_folder == "":
        eval_set_path = args.eval_path
    else:
        eval_set_path = dataset_folder + "/context_dataset_eval.csv"

    tokenizer = RobertaTokenizer.from_pretrained(local_model_path, truncation=True, padding='max_length',
                                                 max_length=max_token_limit)
    model = RobertaForMaskedLM.from_pretrained(local_model_path)

    """print("*** Added the new citations tokens to the tokenizer. Example for acl-200:\n",
          tokenizer.tokenize('Our paper is referencing the paper of Nenkova and Passonneau, 2004'), "\n\n")
    print("*** Another example for peerread:\n",
          tokenizer.tokenize('Our paper is referencing the paper of Gribkoff et al., 2014'), "\n\n")

    print("*** Another example for refseer:\n",
          tokenizer.tokenize('Our paper is referencing the paper of Lecoutre and Boussemart, 2003'), "\n\n")
    print("*** Another example for refseer:\n",
          tokenizer.tokenize('Our paper is referencing the paper of Dawson and Jahanian, 1995'), "\n\n")"""

    print("*** Another example for arxiv:\n",
          tokenizer.tokenize('Our paper is referencing the paper of Łącki et al., 2013'), "\n\n")
    print("*** Another example for arxiv:\n",
          tokenizer.tokenize('Our paper is referencing the paper of Štefančić, 2004'), "\n\n")

    eval_dataset = read_eval_dataset(tokenizer)

    mask_filler = pipeline(
        "fill-mask", model=model, tokenizer=tokenizer, top_k=top_k, device=0
    )
    cit_df_for_test = eval_dataset.to_pandas()

    input_texts_for_test = []
    masked_token_targets = []
    idx_counter = -1
    for _, cit in cit_df_for_test.iterrows():
        idx_counter += 1
        if idx_counter < first_eval_idx:
            continue
        elif idx_counter > last_eval_idx:
            break

        temp_masked_text = cit["masked_cit_context"]

        temp_masked_text = shorten_masked_context_for_limit_if_necessary(temp_masked_text)
        # Ignore lines that have been shortened too much (they have no mask)
        # --> Normally, this situation never happens.
        if temp_masked_text.find("<mask>") == -1:
            continue
        input_texts_for_test.append(temp_masked_text)

        masked_token_targets.append(cit['masked_token_target'])

    print(f"\n======> Number of example contexts to analyze = {len(input_texts_for_test)}\n\n")
    all_preds = mask_filler(input_texts_for_test)

    display_top_k_predictions()
