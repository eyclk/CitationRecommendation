import pandas as pd

preprocessed_contexts_file = "./data_preprocessing_global_info/peerread_global_info_context_50/context_dataset.csv"

max_num_of_examples = 30

dataset = pd.read_csv(preprocessed_contexts_file)
citation_tokens = []
titles = []
abstracts = []

for i in range(len(dataset)):
    temp_row = dataset.iloc[i, :]
    # print(temp_row)
    temp_token = temp_row['masked_token_target']
    if temp_token in citation_tokens:
        continue
    citation_tokens.append(temp_row['masked_token_target'])
    unmasked_context = temp_row['citation_context']
    split_context = unmasked_context.split(" </s> ")
    titles.append(split_context[1])
    abstracts.append(split_context[2])

    if len(titles) == max_num_of_examples:
        break

print(f"Here is a list of {len(titles)} citations. "
      f"Each citation is accompanied by its title and abstract in its own line.\n\n")
for j in range(len(titles)):
    print(f"- Citation name: {citation_tokens[j]} ; Title: {titles[j]} ; Abstract: {abstracts[j]}")
