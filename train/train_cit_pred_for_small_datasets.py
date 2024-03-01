from transformers import RobertaForMaskedLM, Trainer, TrainingArguments, RobertaTokenizer, pipeline
from datasets import Dataset
import pandas as pd
import math
from transformers import DataCollatorWithPadding
from tqdm import tqdm
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--max_token_limit", type=int, default=400, help="Max amount allowed for tokens used for training")
parser.add_argument("--model_name", type=str, help="The name of the new model. This is for saved model and checkpoints")
parser.add_argument("--checkpoints_path", type=str, default="../checkpoints", help="Path of the checkpoints folder")
parser.add_argument("--models_path", type=str, default="../models", help="Path of the models folder")
parser.add_argument("--vocab_additions_path", type=str, help="Path to the additional vocab file of the dataset")
parser.add_argument("--train_path", type=str, help="Path to the training set of the dataset")
parser.add_argument("--eval_path", type=str, help="Path to the evaluation set of the dataset")
parser.add_argument("--dataset_path", type=str, default="", help="Path to the folder of the dataset")
parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs for training")
parser.add_argument("--warmup_steps", type=int, default=500, help="Number of warmup steps for the learning rate")
parser.add_argument("--batch_size", type=int, default=16, help="Batch size for the training and evaluation")
parser.add_argument("--pretrained_model_path", type=str, default="roberta-base", help="Path or name of the pretrained "
                                                                                      "model used at the beginning")
parser.add_argument("--skip_vocab_additions", type=bool, default=False, help="Choose whether to skip vocab additions "
                                                                             "or not")
parser.add_argument("--make_sure_mask_in_middle", type=bool, default=False, help="Run another check on the input "
                                                                                 "contexts and make sure mask tokens "
                                                                                 "are in the middle by cutting from "
                                                                                 "start and end")
parser.add_argument("--output_file", type=str, default="./outputs/train_results.txt", help="Path to file that will "
                                                                                           "contain outputs and "
                                                                                           "results")
parser.add_argument("--tpu_num", type=int, default=-1, help="If set to something other than -1, tpu_num_cores "
                                                            "parameter will be added to TrainingArguments.")
parser.add_argument("--auto_find_batch_size", type=bool, default=False, help="Make this flag True for the Trainer to "
                                                                             "automatically select an appropriate "
                                                                             "batch size")
parser.add_argument("--fp16", type=bool, default=True, help="Make this flag False, so that Trainer can be used with "
                                                            "TPUs")


def add_cit_tokens_to_tokenizer():
    new_token_df = pd.read_csv(additional_vocab_path)
    for _, i in tqdm(new_token_df.iterrows(), total=new_token_df.shape[0]):
        tokenizer.add_tokens(i['additions_to_vocab'])

    model.resize_token_embeddings(len(tokenizer))


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


def make_sure_mask_token_is_in_middle(temp_dataset):
    cit_df = temp_dataset.to_pandas()
    masked_texts = []
    cit_contexts = []
    for _, cit in cit_df.iterrows():
        temp_masked_text = cit["masked_cit_context"]
        masked_texts.append(temp_masked_text)
        cit_contexts.append(cit["citation_context"])

    more_than_400_count = 0

    token_limit = train_max_token_limit
    half_of_limit = int(token_limit / 2)
    fixed_masked_texts = []
    fixed_cit_contexts = []
    for m_idx in range(len(masked_texts)):
        tokenized_id_text = tokenizer.encode(masked_texts[m_idx])[1:-1]
        tokenized_cit_context = tokenizer.encode(cit_contexts[m_idx])[1:-1]

        if len(tokenized_id_text) > train_max_token_limit:
            more_than_400_count += 1
        mask_index = tokenized_id_text.index(50264)  # 50264 is the <mask> token.
        if len(tokenized_id_text) > token_limit+1 and mask_index > half_of_limit:
            new_start_idx = mask_index - half_of_limit
            new_end_idx = mask_index + (half_of_limit-1)
            proper_tokenized_text = tokenized_id_text[new_start_idx:new_end_idx]
            proper_cit_context = tokenized_cit_context[new_start_idx:new_end_idx]
        elif len(tokenized_id_text) > token_limit+1 and mask_index <= half_of_limit:
            proper_tokenized_text = tokenized_id_text[:token_limit]
            proper_cit_context = tokenized_cit_context[:token_limit]
        elif len(tokenized_id_text) <= token_limit+1 and mask_index > half_of_limit:
            proper_tokenized_text = tokenized_id_text[:-1]
            proper_cit_context = tokenized_cit_context[:-1]
        else:
            proper_tokenized_text = tokenized_id_text
            proper_cit_context = tokenized_cit_context

        decoded_masked_text = tokenizer.decode(proper_tokenized_text)
        fixed_masked_texts.append(decoded_masked_text)

        decoded_cit_context = tokenizer.decode(proper_cit_context)
        fixed_cit_contexts.append(decoded_cit_context)

    cit_df['masked_cit_context'] = fixed_masked_texts
    cit_df['citation_context'] = fixed_cit_contexts
    improved_temp_dataset = Dataset.from_pandas(cit_df)

    print("--->> Number of contexts with more than 400 tokens =", more_than_400_count, "\n")
    return improved_temp_dataset


def test_example_input_and_find_hits_at_10_score(val_dataset):
    # ************* EXAMPLE TOP10 PREDICTION TEST BEFORE FINE TUNING ----> ONLY FOR ACL-200
    example_text = "Results We evaluate output summaries using ROUGE-1, ROUGE-2, and ROUGE-SU4 (Lin, 2004), " \
                   "with no stemming and retaining all stopwords. These measures have been shown to correlate best " \
                   "with human judgments in general, but among the automatic measures, ROUGE-1 and ROUGE-2 also " \
                   "correlate best with the Pyramid (<mask>; Nenkova et al., 2007) and Responsiveness manual metrics " \
                   "(Louis and Nenkova, 2009). Moreover, ROUGE-1 has been shown to best reflect human-automatic " \
                   "summary comparisons (Owczarzak et al., 2012). For single concept systems, the results are shown " \
                   "in Table 1, and concept combination system results are given in Table 2."
    mask_filler = pipeline(
        "fill-mask", model=model, tokenizer=tokenizer, top_k=10, device=0
    )
    preds = mask_filler(example_text)
    for pred in preds:
        print(f">>> {pred['sequence']}")
    print("")

    # ************** HITS@10 TEST ON VAL_SET ********************
    cit_df_for_test = val_dataset.to_pandas()

    input_texts_for_test = []
    masked_token_targets = []

    missing_mask_count = 0
    for _, cit in cit_df_for_test.iterrows():
        temp_masked_text = cit["masked_cit_context"]

        if temp_masked_text.find("<mask>") == -1:
            missing_mask_count += 1
            continue

        input_texts_for_test.append(temp_masked_text)

        masked_token_targets.append(cit['masked_token_target'])

    # print(f"\n*************** ===> missing_mask_count = {missing_mask_count}\n\n")  # TEMP INFO PRINTOUT !!!

    all_preds = mask_filler(input_texts_for_test)
    hit_count = 0
    pred_comparison_count = 0
    for j in range(len(all_preds)):
        pred_comparison_count += 1
        temp_preds = all_preds[j]

        target_pred_found = False
        for p in temp_preds:
            if isinstance(p, list):
                for p_in in p:
                    if p_in['token_str'] == masked_token_targets[j]:
                        hit_count += 1
                        target_pred_found = True
            elif p['token_str'] == masked_token_targets[j]:
                hit_count += 1
                target_pred_found = True

            if target_pred_found:
                break

    hit_at_10_metric = hit_count / pred_comparison_count
    print("\n=======>>> Hits@10 measurement value (between 0 and 1) = ", hit_at_10_metric, "\n")
    f_out.write(f"\n=======>>> Hits@10 measurement value (between 0 and 1) = {hit_at_10_metric}\n")


if __name__ == '__main__':
    args = parser.parse_args()

    train_max_token_limit = args.max_token_limit
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
    tpu_core_num = args.tpu_num
    fp16_flag = args.fp16
    if fp16_flag is False:
        print("\n---> Reminder: fp16 flag for TrainingArguments has been given as False. \n\n")
    auto_find_batch_size_flag = args.auto_find_batch_size

    pretrained_model_name_or_path = args.pretrained_model_path

    skip_vocab_additions = args.skip_vocab_additions
    make_sure_mask_in_middle_flag = args.make_sure_mask_in_middle

    tokenizer = RobertaTokenizer.from_pretrained(pretrained_model_name_or_path, truncation=True, padding='max_length',
                                                 max_length=train_max_token_limit)
    model = RobertaForMaskedLM.from_pretrained(pretrained_model_name_or_path)

    f_out = open(args.output_file, "w")

    # --------------------------------------------------------------------------------------------
    if not skip_vocab_additions:
        add_cit_tokens_to_tokenizer()

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

    if make_sure_mask_in_middle_flag:
        train_set = make_sure_mask_token_is_in_middle(train_set)
        val_set = make_sure_mask_token_is_in_middle(val_set)
        print("\n*** Train and Val sets are made sure to have appropriate number of tokens and proper mask "
              "placements.\n\n")

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=checkpoints_location,
        overwrite_output_dir=True,
        evaluation_strategy="epoch",
        weight_decay=0.01,
        # per_device_train_batch_size=train_and_eval_batch_sizes,
        # per_device_eval_batch_size=train_and_eval_batch_sizes,
        push_to_hub=False,
        fp16=fp16_flag,
        logging_strategy="epoch",
        num_train_epochs=num_epochs,
        warmup_steps=warmup_steps,
        load_best_model_at_end=False,
        save_strategy="epoch",
        save_total_limit=5
    )

    if auto_find_batch_size_flag is True:
        training_args.auto_find_batch_size = True
    else:
        training_args.per_device_train_batch_size = train_and_eval_batch_sizes
        training_args.per_device_eval_batch_size = train_and_eval_batch_sizes

    if tpu_core_num > 0:
        training_args.tpu_num_cores = tpu_core_num

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

    # test_example_input_and_find_hits_at_10_score(val_set)

    trainer.train()
    trainer.save_model(model_save_location)

    eval_results = trainer.evaluate()
    print(f"\n*****************\n======>> Perplexity after fine-tuning: {math.exp(eval_results['eval_loss']):.2f}\n\n")
    f_out.write(f"======>> Perplexity after fine-tuning: {math.exp(eval_results['eval_loss']):.2f}\n\n")

    test_example_input_and_find_hits_at_10_score(val_set)

    f_out.close()
