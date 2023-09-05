from transformers import RobertaForMaskedLM, Trainer, TrainingArguments, RobertaTokenizer, pipeline
from datasets import Dataset
import pandas as pd
import math
from transformers import DataCollatorWithPadding
from tqdm import tqdm


eval_max_token_limit = 400
train_max_token_limit = 400
custom_model_name = "cit_pred_base"
additional_vocab_path = "./cit_data/additions_to_vocab.csv"
cit_dataset_path = "./cit_data/context_only_dataset.csv"

num_epochs = 200
warmup_steps = 1000
train_and_eval_batch_sizes = 32

tokenizer = RobertaTokenizer.from_pretrained("roberta-base", truncation=True, padding=True,
                                             max_length=train_max_token_limit)
model = RobertaForMaskedLM.from_pretrained("roberta-base")


def add_cit_tokens_to_tokenizer():
    new_token_df = pd.read_csv(additional_vocab_path)
    for _, i in tqdm(new_token_df.iterrows(), total=new_token_df.shape[0]):
        tokenizer.add_tokens(i['additions_to_vocab'])

    model.resize_token_embeddings(len(tokenizer))  # , pad_to_multiple_of=8


def tokenizer_function(tknizer, inp_data, col_name):  # ************** TOKEN LIMIT CAN ALSO BE INCREASED LATER!!! 350
    return tknizer(inp_data[col_name], truncation=True, padding=True, max_length=train_max_token_limit)


def prepare_data():
    cit_df = pd.read_csv(cit_dataset_path)
    input_texts = []
    label_texts = []
    masked_token_targets = []

    # c = 0
    for _, i in cit_df.iterrows():
        input_texts.append(i['masked_cit_context'])
        label_texts.append(i['citation_context'])
        masked_token_targets.append(i['masked_token_target'])

        # FOR TESTING PURPOSES ***
        """c += 1
        if c >= 5000:
            break"""

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

    raw_and_tokenized_data = raw_and_tokenized_data.train_test_split(test_size=0.2, seed=42)
    train_dataset = raw_and_tokenized_data["train"]
    val_dataset = raw_and_tokenized_data["test"]
    return train_dataset, val_dataset


def check_if_masked_context_exceed_token_limit(masked_text):
    tokenized_text = tokenizer.tokenize(masked_text)
    if len(tokenized_text) > eval_max_token_limit:  # Ignore texts with more than 500 tokens.
        return True
    return False


def test_example_input_and_find_hits_at_10_score(val_dataset):
    # ************* EXAMPLE TOP10 PREDICTION TEST BEFORE FINE TUNING
    example_text = "Results We evaluate output summaries using ROUGE-1, ROUGE-2, and ROUGE-SU4 (Lin, 2004), with no stemming and retaining all stopwords. These measures have been shown to correlate best with human judgments in general, but among the automatic measures, ROUGE-1 and ROUGE-2 also correlate best with the Pyramid (<mask>; Nenkova et al., 2007) and Responsiveness manual metrics (Louis and Nenkova, 2009). Moreover, ROUGE-1 has been shown to best reflect human-automatic summary comparisons (Owczarzak et al., 2012). For single concept systems, the results are shown in Table 1, and concept combination system results are given in Table 2."
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
    for _, cit in cit_df_for_test.iterrows():
        temp_masked_text = cit["masked_cit_context"]
        if check_if_masked_context_exceed_token_limit(temp_masked_text):
            continue
        input_texts_for_test.append(temp_masked_text)

        masked_token_targets.append(cit['masked_token_target'])

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


if __name__ == '__main__':

    add_cit_tokens_to_tokenizer()
    print("*** Added the new citations tokens to the tokenizer. Example:\n",
          tokenizer.tokenize('Our paper is referencing the paper of Nenkova and Passonneau, 2004'), "\n\n")

    train_set, val_set = prepare_data()
    print("\n\n*** Train and Val sets are read and split into proper CustomCitDataset classes.")

    test_example_input_and_find_hits_at_10_score(val_set)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=f"./checkpoints/{custom_model_name}",
        overwrite_output_dir=True,
        evaluation_strategy="epoch",
        # learning_rate=2e-5,
        weight_decay=0.01,
        per_device_train_batch_size=train_and_eval_batch_sizes,
        per_device_eval_batch_size=train_and_eval_batch_sizes,
        push_to_hub=False,
        fp16=True,
        # logging_steps=logging_steps,
        num_train_epochs=num_epochs,
        warmup_steps=warmup_steps
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
    eval_results = trainer.evaluate()  # eval_dataset is being used as the test_data for now.
    print(f"\n======>> Perplexity before finetuning: {math.exp(eval_results['eval_loss']):.2f}\n")

    trainer.train()
    trainer.save_model(f'./models/{custom_model_name}')

    eval_results = trainer.evaluate()
    print(f"\n*****************\n======>> Perplexity after finetuning: {math.exp(eval_results['eval_loss']):.2f}\n\n")

    test_example_input_and_find_hits_at_10_score(val_set)
