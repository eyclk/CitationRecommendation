from transformers import Trainer, TrainingArguments, pipeline, BartTokenizer, BartForConditionalGeneration
from datasets import Dataset
import pandas as pd
import math
from transformers import DataCollatorWithPadding
# from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--max_token_limit", type=int, default=400, help="Max amount allowed for tokens used for training "
                                                                     "and evaluation")
parser.add_argument("--model_name", type=str, help="The name of the new model. This is for saved model and checkpoints")
parser.add_argument("--checkpoints_path", type=str, default="../../checkpoints", help="Path of the checkpoints folder")
parser.add_argument("--models_path", type=str, default="../../models", help="Path of the models folder")
parser.add_argument("--vocab_additions_path", type=str, help="Path to the additional vocab file of the dataset")
parser.add_argument("--train_path", type=str, help="Path to the training set of the dataset")
parser.add_argument("--eval_path", type=str, help="Path to the evaluation set of the dataset")
parser.add_argument("--dataset_path", type=str, default="", help="Path to the folder of the dataset")
parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs for training")
parser.add_argument("--warmup_steps", type=int, default=500, help="Number of warmup steps for the learning rate")
parser.add_argument("--batch_size", type=int, default=16, help="Batch size for the training and evaluation")
parser.add_argument("--pretrained_model_path", type=str, default="facebook/bart-base", help="Path or name of the "
                                                                                            "pretrained model used at "
                                                                                            "the beginning")
# parser.add_argument("--skip_vocab_additions", type=bool, default=False, help="Choose whether to skip vocab additions "
#                                                                           "or not")
parser.add_argument("--output_file", type=str, default="../outputs/train_results.txt", help="Path to file that will "
                                                                                            "contain outputs and "
                                                                                            "results")

"""def add_cit_tokens_to_tokenizer():
    new_token_df = pd.read_csv(additional_vocab_path)
    for _, i in tqdm(new_token_df.iterrows(), total=new_token_df.shape[0]):
        tokenizer.add_tokens(i['additions_to_vocab'])

    model.resize_token_embeddings(len(tokenizer))"""


def tokenizer_function(tknizer, inp_data, col_name):
    return tknizer(inp_data[col_name], truncation=True, padding='max_length', max_length=train_max_token_limit)


def read_dataset_csv_files(train_or_eval="train"):
    if train_or_eval == "train":
        temp_path = train_dataset_path
    else:
        temp_path = eval_dataset_path

    cit_df = pd.read_csv(temp_path)
    input_texts = []
    label_texts = []
    masked_token_targets = []

    for _, i in cit_df.iterrows():
        input_texts.append(i['masked_cit_context'])
        label_texts.append(i['citation_context'])
        masked_token_targets.append(i['masked_token_target'])

    df_text_list = pd.DataFrame(input_texts, columns=['input_ids'])
    data_input_ids = Dataset.from_pandas(df_text_list)
    tokenized_input_ids = data_input_ids.map(
        lambda batch: tokenizer_function(tokenizer, batch, 'input_ids'), batched=True)

    df_label_list = pd.DataFrame(label_texts, columns=['labels'])
    data_labels = Dataset.from_pandas(df_label_list)
    tokenized_labels = data_labels.map(
        lambda batch: tokenizer_function(tokenizer, batch, 'labels'), batched=True)

    tokenized_data = tokenized_input_ids.add_column('labels', tokenized_labels['input_ids'])

    raw_and_tokenized_data = tokenized_data.add_column('masked_cit_context', input_texts)
    raw_and_tokenized_data = raw_and_tokenized_data.add_column('citation_context', label_texts)
    raw_and_tokenized_data = raw_and_tokenized_data.add_column('masked_token_target', masked_token_targets)

    return raw_and_tokenized_data


def prepare_data():
    train_dataset = read_dataset_csv_files(train_or_eval="train")
    eval_dataset = read_dataset_csv_files(train_or_eval="eval")

    return train_dataset, eval_dataset


def shorten_masked_context_for_limit_if_necessary(masked_text):
    tokenized_text = tokenizer.tokenize(masked_text)
    if len(tokenized_text) > eval_max_token_limit:  # Shorten texts with more than the max limit.
        exceeding_char_count = 5  # Always start with 5 extra characters just in case.
        for i in range(eval_max_token_limit - 1, len(tokenized_text)):
            exceeding_char_count += len(tokenized_text[i])
        shortened_text = masked_text[:-exceeding_char_count]
        return shortened_text
    return masked_text


def find_exact_match_score(val_dataset):
    mask_filler = pipeline(
        "fill-mask", model=model, tokenizer=tokenizer, top_k=3, device=0,
    )

    example_text = "n techniques, limits, improvement schemenerally, a well-designed language model makes a critical difference in various natural language processing  tasks, like speech recognition , machine translation <mask> , semantic extraction  and etc. guage modeling , therefore, has been the research focus in NLP field all the time, and a large number of sound research results have been published in the past decades."
    example_ground_truth = "n techniques, limits, improvement schemenerally, a well-designed language model makes a critical difference in various natural language processing  tasks, like speech recognition , machine translation Cho et al., 2014 , semantic extraction  and etc. guage modeling , therefore, has been the research focus in NLP field all the time, and a large number of sound research results have been published in the past decades."

    preds = mask_filler(example_text)
    for pred in preds:
        print(f">>> {pred['sequence']}\n")
    print("\n")

    """batch = tokenizer(example_text, return_tensors="pt", truncation=True, padding='max_length',
                      max_length=train_max_token_limit).to("cuda")
    generated_ids = model.generate(batch["input_ids"], max_length=eval_max_token_limit)  # , max_new_tokens=20"""
    #  , num_beams=5, num_return_sequences=3, temperature=1.5, do_sample=True

    """predicted_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    if predicted_text[0] == example_ground_truth:
        print("\n------>Examples text's mask has been filled correctly!!!\n\n")
    else:
        print(f"\n------>Incorrect example text prediction: '{predicted_text}' \n\n")"""

    # ************** HITS@10 TEST ON VAL_SET ********************
    cit_df_for_test = val_dataset.to_pandas()

    input_texts_for_test = []
    ground_truth_contexts = []
    for _, cit in cit_df_for_test.iterrows():
        temp_masked_text = cit["masked_cit_context"]

        temp_masked_text = shorten_masked_context_for_limit_if_necessary(temp_masked_text)
        if temp_masked_text.find("<mask>") == -1:
            continue
        input_texts_for_test.append(temp_masked_text)

        ground_truth_contexts.append(cit['citation_context'])

    # all_preds = mask_filler(input_texts_for_test)
    eval_batch = tokenizer(input_texts_for_test, return_tensors="pt", truncation=True, padding='max_length',
                           max_length=train_max_token_limit).to("cuda")
    #  , num_beams=15, num_return_sequences=10, temperature=1.5

    # print(f"\n********* eval_batch ---->> {eval_batch}\n")

    all_preds = model(eval_batch["input_ids"])  # ["input_ids"] , max_length=eval_max_token_limit

    print(f"\n********* shape of all_preds before decode ---->> {all_preds.shape}\n")
    all_preds = tokenizer.batch_decode(all_preds, skip_special_tokens=True)
    print(f"********* shape of all_preds after decode ---->> {all_preds.shape}\n")

    hit_count = 0
    pred_comparison_count = 0
    """for j in range(len(all_preds)):
        pred_comparison_count += 1
        temp_preds = all_preds[j]

        target_pred_found = False
        for p in temp_preds:
            if isinstance(p, list):
                for p_in in p:
                    if p_in['sequence'] == ground_truth_contexts[j]:
                        hit_count += 1
                        target_pred_found = True
            elif p['sequence'] == ground_truth_contexts[j]:
                hit_count += 1
                target_pred_found = True

            if target_pred_found:
                break"""
    for j in range(len(all_preds)):
        pred_comparison_count += 1
        temp_preds = all_preds[j]
        for p in temp_preds:
            if p == ground_truth_contexts[j]:
                hit_count += 1
                break

    hit_at_10_metric = hit_count / pred_comparison_count
    print("\n=======>>> Hits@10 measurement value (between 0 and 1) = ", hit_at_10_metric, "\n")
    f_out.write(f"\n=======>>> Hits@10 measurement value (between 0 and 1) = {hit_at_10_metric}\n")


if __name__ == '__main__':
    args = parser.parse_args()

    train_max_token_limit = args.max_token_limit
    eval_max_token_limit = args.max_token_limit
    custom_model_name = args.model_name
    checkpoints_location = f"{args.checkpoints_path}/{custom_model_name}"
    model_save_location = f"{args.models_path}/{custom_model_name}"

    dataset_folder = args.dataset_path
    if dataset_folder == "":
        additional_vocab_path = args.vocab_additions_path
        train_dataset_path = args.train_path
        eval_dataset_path = args.eval_path
    else:
        additional_vocab_path = dataset_folder + "/additions_to_vocab.csv"
        train_dataset_path = dataset_folder + "/context_dataset_train.csv"
        eval_dataset_path = dataset_folder + "/context_dataset_eval.csv"

    num_epochs = args.num_epochs
    warmup_steps = args.warmup_steps
    train_and_eval_batch_sizes = args.batch_size

    pretrained_model_name_or_path = args.pretrained_model_path

    # skip_vocab_additions = args.skip_vocab_additions

    tokenizer = BartTokenizer.from_pretrained(pretrained_model_name_or_path, truncation=True, padding='max_length',
                                              max_length=train_max_token_limit)
    model = BartForConditionalGeneration.from_pretrained(pretrained_model_name_or_path)

    f_out = open(args.output_file, "w")

    # --------------------------------------------------------------------------------------------
    """if not skip_vocab_additions:
        add_cit_tokens_to_tokenizer()"""

    print("*** Added the new citations tokens to the tokenizer. Example for acl-200:\n",
          tokenizer.tokenize('Our paper is referencing the paper of Nenkova and Passonneau, 2004'), "\n\n")
    print("*** Another example for peerread:\n",
          tokenizer.tokenize('Our paper is referencing the paper of Gribkoff et al., 2014'), "\n\n")
    print("*** Another example for refseer:\n",
          tokenizer.tokenize('Our paper is referencing the paper of Lecoutre and Boussemart, 2003'), "\n\n")
    print("*** Another example for arxiv:\n",
          tokenizer.tokenize('Our paper is referencing the paper of Fishman et al., 2009'), "\n\n")

    train_set, val_set = prepare_data()
    print("\n\n*** Train and Val sets are read and split into proper CustomCitDataset classes.")

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=checkpoints_location,
        overwrite_output_dir=True,
        evaluation_strategy="epoch",
        weight_decay=0.01,
        per_device_train_batch_size=train_and_eval_batch_sizes,
        per_device_eval_batch_size=train_and_eval_batch_sizes,
        push_to_hub=False,
        fp16=True,
        logging_strategy="epoch",
        num_train_epochs=num_epochs,
        warmup_steps=warmup_steps,
        load_best_model_at_end=False,
        save_strategy="epoch",
        save_total_limit=5
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=val_set,
        data_collator=data_collator,
        tokenizer=tokenizer
    )

    # Evaluate and acquire perplexity score
    # eval_results = trainer.evaluate()  # eval_dataset is being used as the test_data for now.
    # print(f"\n======>> Perplexity before fine-tuning: {math.exp(eval_results['eval_loss']):.2f}\n")

    find_exact_match_score(val_set)  # TEMP !!!!

    trainer.train()
    trainer.save_model(model_save_location)

    eval_results = trainer.evaluate()
    print(f"\n*****************\n======>> Perplexity after fine-tuning: {math.exp(eval_results['eval_loss']):.2f}\n\n")
    f_out.write(f"======>> Perplexity after fine-tuning: {math.exp(eval_results['eval_loss']):.2f}\n\n")

    find_exact_match_score(val_set)

    f_out.close()
