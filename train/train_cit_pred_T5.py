from typing import List, Any
from datasets import DatasetDict, Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments, DataCollatorWithPadding
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
parser.add_argument("--pretrained_model_path", type=str, default="google-t5/t5-base", help="Path or name of "
                                                                                           "the pretrained model used "
                                                                                           "at the beginning")
parser.add_argument("--auto_find_batch_size", type=bool, default=False, help="Make this flag True for the Trainer to "
                                                                             "automatically select an appropriate "
                                                                             "batch size")
parser.add_argument("--skip_training", type=bool, default=False, help="Skips training and directly perform evaluation")


# Preprocessing function
def preprocess_function(examples):
    inputs = [example.replace("<mask>", "<extra_id_0>", 1).replace("<mask>", "")
              for example in examples["masked_cit_context"]]
    targets = [example for example in examples["citation_context"]]

    model_inputs = tokenizer(inputs, max_length=max_token_limit, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=max_token_limit, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def read_dataset():
    train_df = pd.read_csv(train_dataset_path)
    train_set = []

    for _, i in train_df.iterrows():
        temp_dict = {"masked_cit_context": i['masked_cit_context'], "citation_context": i['citation_context'],
                     "masked_token_target": i['masked_token_target']}

        train_set.append(temp_dict)

    eval_df = pd.read_csv(eval_dataset_path)
    eval_set = []

    for _, i in eval_df.iterrows():
        temp_dict = {"masked_cit_context": i['masked_cit_context'], "citation_context": i['citation_context'],
                     "masked_token_target": i['masked_token_target']}

        eval_set.append(temp_dict)

    return train_set, eval_set


def fill_mask(sentence, num_predictions=30):
    input_ids = tokenizer.encode(sentence.replace("<mask>", "<extra_id_0>").replace("<mask>", ""),
                                 return_tensors="pt", max_length=max_token_limit, truncation=True,
                                 padding="max_length").to("cuda")

    outputs = model.generate(
        input_ids,
        max_new_tokens=50,
        do_sample=True,
        num_return_sequences=num_predictions,
        top_k=50,  # Consider only the top 50 words by predicted probability
        top_p=0.95,  # Consider only the top 95% of words by cumulative probability
        temperature=0.7,
        # Lower temperature results in more focused predictions, higher temperature in more random predictions
        # eos_token_id=tokenizer.convert_tokens_to_ids("<extra_id_1>")
    )

    predictions = []
    for output in outputs:
        decoded_output = tokenizer.decode(output, skip_special_tokens=False)
        start_token = "<extra_id_0>"
        end_token = "<extra_id_1>"
        start_index = decoded_output.find(start_token) + len(start_token)
        end_index = decoded_output.find(end_token, start_index)
        temp_prediction = decoded_output[start_index:end_index].strip()
        predictions.append(temp_prediction)

    # Get unique predictions
    unique_predictions: List[Any] = list(dict.fromkeys(predictions))  # Remove duplicates while preserving order

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
                print(f"\nGround truth citation: {ground_truth}\nCorrect Pred =====>> {p}\n")
                return True
    elif "et al" in ground_truth:
        truth_tokens = ground_truth.replace(" et al.,", "").split()
        for p in predictions:
            if truth_tokens[0] in p and truth_tokens[1] in p:
                print(f"\nGround truth citation: {ground_truth}\nCorrect Pred =====>> {p}\n")
                return True
    else:
        truth_tokens = ground_truth.replace(",", "").split()
        for p in predictions:
            if truth_tokens[0] in p and truth_tokens[1] in p:
                print(f"\nGround truth citation: {ground_truth}\nCorrect Pred =====>> {p}\n")
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
        is_pred_correct = compare_pred_with_correct_value(temp_predictions, target_token)
        if is_pred_correct:
            hit_count += 1

    hit_at_10_metric = hit_count / pred_comparison_count
    print("\n=======>>> Hits@10 measurement value (between 0 and 1) = ", hit_at_10_metric, "\n")


if __name__ == '__main__':
    args = parser.parse_args()

    max_token_limit = args.max_token_limit
    # eval_max_token_limit = args.max_token_limit
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

    # Initialize the tokenizer
    tokenizer = T5Tokenizer.from_pretrained(pretrained_model_name_or_path, legacy=True, truncation=True,
                                            padding='max_length', model_max_length=max_token_limit)

    # Set up the model
    model = T5ForConditionalGeneration.from_pretrained(pretrained_model_name_or_path)

    # Example data to view dataset structure
    """data = {
        "train": [
            {"input": "The oil spill in the gulf is <mask>.",
             "target": "The oil spill in the gulf is due to the oil tankers."},
            {"input": "Climate change leads to <mask>.",
             "target": "Climate change leads to severe weather patterns."},
            # Add more examples...
        ],
        "validation": [
            {"input": "The new policy is <mask>.",
             "target": "The new policy is under review."},
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

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

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
    print(f"\n*****************\n======>> Perplexity after fine-tuning: {math.exp(eval_results['eval_loss']):.2f}\n\n")

    calc_hits_at_10_score(eval_dataset)
