from typing import List, Any
from datasets import DatasetDict, Dataset
from transformers import (BartForConditionalGeneration, BartTokenizer, Trainer, TrainingArguments,
                          BartConfig, GenerationConfig, DataCollatorForSeq2Seq)
import pandas as pd
import argparse
import math
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--max_token_limit", type=int, default=400, help="Max amount allowed for tokens used for training "
                                                                     "and evaluation")
parser.add_argument("--model_name", type=str, help="The name of the new model. This is for saved model and checkpoints")
parser.add_argument("--checkpoints_path", type=str, default="../checkpoints", help="Path of the checkpoints folder")
parser.add_argument("--models_path", type=str, default="../models", help="Path of the models folder")
parser.add_argument("--dataset_path", type=str, help="Path to the folder of the dataset")
parser.add_argument("--num_epochs", type=int, default=5, help="Number of epochs for training")
parser.add_argument("--warmup_steps", type=int, default=500, help="Number of warmup steps for the learning rate")
parser.add_argument("--batch_size", type=int, default=16, help="Batch size for the training and evaluation")
parser.add_argument("--pretrained_model_path", type=str, default="facebook/bart-base", help="Path or name "
                                                                                            "of the pretrained model "
                                                                                            "used at the beginning")
parser.add_argument("--auto_find_batch_size", type=bool, default=False, help="Make this flag True for the Trainer to "
                                                                             "automatically select an appropriate "
                                                                             "batch size")
parser.add_argument("--skip_training", type=bool, default=False, help="Skips training and directly perform evaluation")


# Preprocessing function
def preprocess_function(examples):
    inputs = [example for example in examples["masked_cit_context"]]
    # inputs = [example.replace("<mask>", "<extra_id_0>", 1).replace("<mask>", "").replace("<extra_id_0>", "<mask>")
    #           for example in examples["masked_cit_context"]]
    targets = [example for example in examples["masked_token_target"]]

    model_inputs = tokenizer(inputs, max_length=max_token_limit, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=max_token_limit, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def read_dataset():
    train_df = pd.read_csv(train_dataset_path)
    train_set = []

    for _, i in train_df.iterrows():
        temp_masked_context = (("CONTEXT: " + i['masked_cit_context'].replace("<mask>", "<extra_id_0>", 1)
                               .replace("<mask>", "").replace("<extra_id_0>", "TARGETCITE"))
                               + " </s> TARGETCITE: <mask>")
        temp_token_target = i['masked_token_target']
        temp_dict = {"masked_cit_context": temp_masked_context, "citation_context": i['citation_context'],
                     "masked_token_target": temp_token_target}

        train_set.append(temp_dict)

    eval_df = pd.read_csv(eval_dataset_path)
    eval_set = []

    for _, i in eval_df.iterrows():
        temp_masked_context = (("CONTEXT: various natural language processing  tasks, like speech recognition , machine"
                                " translation TARGETCITE , semantic extraction  and etc. gauge modeling , therefore, "
                                "has been the research focus in NLP field </s> TARGETCITE: Cho et al., 2014 </s> "
                                "CONTEXT: " + i['masked_cit_context'].replace("<mask>", "<extra_id_0>", 1)
                                .replace("<mask>", "").replace("<extra_id_0>", "TARGETCITE"))
                               + " </s> TARGETCITE: <mask>")
        temp_token_target = i['masked_token_target']
        temp_dict = {"masked_cit_context": temp_masked_context, "citation_context": i['citation_context'],
                     "masked_token_target": temp_token_target}

        eval_set.append(temp_dict)

    return train_set, eval_set


def fill_mask(sentence):
    # .replace("<mask>", "<extra_id_0>").replace("<mask>", "").replace("<extra_id_0>", "<mask>")
    input_ids = tokenizer.encode(sentence,
                                 return_tensors="pt", max_length=max_token_limit, truncation=True,
                                 padding="max_length").to("cuda")

    outputs = model.generate(
        input_ids,
        generation_config=cit_generation_config
    )

    predictions = []
    for output in outputs:
        decoded_output = tokenizer.decode(output, skip_special_tokens=True)
        temp_prediction = decoded_output.strip()
        predictions.append(temp_prediction)

    # Get unique predictions
    unique_predictions: List[Any] = list(dict.fromkeys(predictions))  # Remove duplicates while preserving order

    # Print the top 10 predictions
    # for i, pred in enumerate(unique_predictions, 1):
    #     print(f"Prediction {i}: {pred} \n\n")

    last_item_of_predictions = unique_predictions[-1]
    while len(unique_predictions) < 10:
        unique_predictions.append(last_item_of_predictions)

    # Return the top 10 unique predictions, or fewer if not enough unique ones are available
    return unique_predictions[:10]


def compare_pred_with_correct_value(predictions, ground_truth):
    if "and" in ground_truth:
        truth_tokens = ground_truth.replace(" and ", ", ").replace(",", "").split()
        for p in predictions:
            if truth_tokens[0] in p and truth_tokens[1] in p and truth_tokens[2] in p:
                # print(f"\n---> Ground truth citation: {ground_truth}\nCorrect Pred =====>> {p}\n")
                return True
    elif "et al" in ground_truth:
        truth_tokens = ground_truth.replace(" et al.,", "").split()
        for p in predictions:
            if truth_tokens[0] in p and truth_tokens[1] in p:
                # print(f"\n---> Ground truth citation: {ground_truth}\nCorrect Pred =====>> {p}\n")
                return True
    else:
        truth_tokens = ground_truth.replace(",", "").split()
        for p in predictions:
            if truth_tokens[0] in p and truth_tokens[1] in p:
                # print(f"\n---> Ground truth citation: {ground_truth}\nCorrect Pred =====>> {p}\n")
                return True
    return False


def calc_hits_at_10_score(val_dataset):
    hit_count = 0
    pred_comparison_count = 0
    for e in tqdm(val_dataset):
        pred_comparison_count += 1
        masked_cit_context = e["masked_cit_context"]
        target_token = e["masked_token_target"]

        temp_predictions = fill_mask(masked_cit_context)

        # print(f"\n----------------------->>> Ground truth cit = {target_token}\n\n")

        is_pred_correct = compare_pred_with_correct_value(temp_predictions, target_token)
        if is_pred_correct:
            hit_count += 1

    hit_at_10_metric = hit_count / pred_comparison_count
    print("\n=======>>> Hits@10 measurement value (between 0 and 1) = ", hit_at_10_metric, "\n")


if __name__ == '__main__':
    args = parser.parse_args()

    max_token_limit = args.max_token_limit
    custom_model_name = args.model_name
    checkpoints_location = f"{args.checkpoints_path}/{custom_model_name}"
    model_save_location = f"{args.models_path}/{custom_model_name}"

    dataset_folder = args.dataset_path
    train_dataset_path = dataset_folder + "/context_dataset_train.csv"
    eval_dataset_path = dataset_folder + "/context_dataset_eval.csv"

    num_epochs = args.num_epochs

    warmup_steps = args.warmup_steps
    train_and_eval_batch_sizes = args.batch_size

    auto_find_batch_size_flag = args.auto_find_batch_size

    pretrained_model_name_or_path = args.pretrained_model_path

    skip_training = args.skip_training

    # Initialize the config
    config = BartConfig.from_pretrained(pretrained_model_name_or_path, attention_dropout=0.123)

    # Initialize the tokenizer
    tokenizer = BartTokenizer.from_pretrained(pretrained_model_name_or_path, truncation=True,
                                              padding='max_length', model_max_length=max_token_limit)

    # Set up the model
    model = BartForConditionalGeneration.from_pretrained(pretrained_model_name_or_path, config=config)

    cit_generation_config = GenerationConfig.from_model_config(model.config)

    cit_generation_config.max_new_tokens = 25
    cit_generation_config.do_sample = False
    cit_generation_config.top_k = 50
    # cit_generation_config.eos_token_id=model.config.eos_token_id
    cit_generation_config.num_return_sequences = 20
    # cit_generation_config.top_p = 0.95
    # cit_generation_config.temperature = 0.7
    cit_generation_config.early_stopping = False
    cit_generation_config.num_beams = 20
    cit_generation_config.forced_bos_token_id = 0

    # VERY IMPORTANT PARAMETERS !!!!!!!!!!!!!!
    # cit_generation_config.repetition_penalty = 1.2
    cit_generation_config.num_beam_groups = 5
    cit_generation_config.diversity_penalty = 1.5

    # Example data to view dataset structure
    """data = {
        "train": [
            {"input": "CONTEXT: models are trained end-to-end using backpropagation
             and mini-batched Adam  TARGETCITE SGD. We use dropout regularization </s> TARGETCITE: <mask>",
             "target": "Kingma and Ba, 2014"},
            # Add more examples...
        ],
        "validation": [
            {"input": "CONTEXT: various natural language processing  tasks, like speech recognition , machine 
            translation TARGETCITE , semantic extraction  and etc. gauge modeling , therefore, has been the research 
            focus in NLP field </s> TARGETCITE: Cho et al., 2014 
            </s> CONTEXT: experimented with several combination methods , regularization factors, and pre-trained word 
            embeddings TARGETCITE . This time, we used cosine similarity  to separate related </s> TARGETCITE: <mask>",
             "target": "Mikolov et al., 2013"},
            # Add more examples...
        ]
    }"""

    train_dataset, eval_dataset = read_dataset()

    data = {
        "train": train_dataset,
        "eval": eval_dataset
    }

    # Convert to Dataset
    train_dataset = Dataset.from_pandas(pd.DataFrame(data["train"]))
    validation_dataset = Dataset.from_pandas(pd.DataFrame(data["eval"]))

    dataset = DatasetDict({
        "train": train_dataset,
        "eval": validation_dataset
    })

    # Preprocess the datasets
    tokenized_datasets = dataset.map(preprocess_function, batched=True)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    training_args = TrainingArguments(
        output_dir=checkpoints_location,
        overwrite_output_dir=True,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        logging_strategy="epoch",
        warmup_steps=warmup_steps,
        save_strategy="epoch",
        save_total_limit=4
    )

    if auto_find_batch_size_flag is True:
        training_args.auto_find_batch_size = True
    else:
        training_args.per_device_train_batch_size = train_and_eval_batch_sizes
        training_args.per_device_eval_batch_size = train_and_eval_batch_sizes

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["eval"],
        data_collator=data_collator,
        tokenizer=tokenizer
    )

    if not skip_training:
        trainer.train()

        trainer.save_model(model_save_location)
        tokenizer.save_pretrained(model_save_location)

    eval_results = trainer.evaluate()
    print(f"\n*****************\n======>> Eval loss after fine-tuning: {eval_results['eval_loss']}\n"
          f"======>> Perplexity after fine-tuning: {math.exp(eval_results['eval_loss']):.2f}\n\n")

    calc_hits_at_10_score(eval_dataset)
