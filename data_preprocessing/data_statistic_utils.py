import pandas as pd
import matplotlib.pyplot as plt
import statistics
import json

dataset_name = "arxiv300k_neg_sampling"
all_contexts_file = f"./{dataset_name}/context_dataset.csv"


def count_ref_appearances():
    contexts_df = pd.read_csv(all_contexts_file)
    appearance_count_dict = {}
    for i in contexts_df.iterrows():
        if i[1]["masked_token_target"] not in appearance_count_dict.keys():
            appearance_count_dict[i[1]["masked_token_target"]] = 1
        else:
            appearance_count_dict[i[1]["masked_token_target"]] += 1

    return appearance_count_dict


def find_min_and_max_ref_appearances(appearance_count_dict):
    key_with_max_ref_count = max(appearance_count_dict, key=appearance_count_dict.get)
    key_with_min_ref_count = min(appearance_count_dict, key=appearance_count_dict.get)

    max_ref_count = appearance_count_dict[key_with_max_ref_count]
    min_ref_count = appearance_count_dict[key_with_min_ref_count]

    print("Max ref count =", max_ref_count, "\n")
    print("References with the max amount =", key_with_max_ref_count, "\n")

    print("Min ref count =", min_ref_count, "\n")  # There are multiple refs with just 1 appearance!!!
    print("References with the min amount =", key_with_min_ref_count, "\n")  # An example with minimum appearance


def find_all_appearance_counts(appearance_count_dict):
    all_appearance_counts_lst = []
    for i in appearance_count_dict.keys():
        all_appearance_counts_lst.append(appearance_count_dict[i])

    return all_appearance_counts_lst


def draw_histogram_of_all_cit_per_context(all_appearance_counts_lst):
    plt.hist(all_appearance_counts_lst, bins=25)
    plt.title(f"Histogram of citations per contexts - Complete version - {dataset_name}")
    plt.show()


def draw_histogram_of_partial_cit_per_context(appearance_count_dict):
    all_appearance_counts_lst = []
    for i in appearance_count_dict.keys():
        if appearance_count_dict[i] > 75:
            continue
        all_appearance_counts_lst.append(appearance_count_dict[i])

    plt.hist(all_appearance_counts_lst, bins=15)
    plt.title(f"Histogram of citations per contexts - Partial version (Less than 75 count) - {dataset_name}")
    plt.show()


def find_how_many_cites_have_more_than_75_appearances_and_less_than_7(appearance_count_dict):
    more_than_75_count = 0
    less_than_7_count = 0
    only_1_count = 0
    for i in appearance_count_dict.keys():
        if appearance_count_dict[i] > 75:
            more_than_75_count += 1
        elif appearance_count_dict[i] < 7:
            less_than_7_count += 1
        if appearance_count_dict[i] == 1:
            only_1_count += 1
    print("\n--> Number of citations that have more than 75 appearances inside the contexts of the dataset =",
          more_than_75_count)
    print("\n--> Number of citations that have less than 7 appearances inside the contexts of the dataset =",
          less_than_7_count)
    print("\n--> Number of citations that have only 1 appearance inside the contexts of the dataset =",
          only_1_count)


def calculate_average_and_median_of_cit_per_ref(appearance_counts_list):
    avg = statistics.mean(appearance_counts_list)
    median = statistics.median(appearance_counts_list)

    print("\n===> Average of all citation counts per references of the dataset =", avg)
    print("\n===> Median of all citation counts per references of the dataset =", median)
    return avg, median


def create_appearance_count_frequency_dict(appearance_counts_list):
    appearance_count_frequency_dict = {}
    for i in appearance_counts_list:
        if i not in appearance_count_frequency_dict.keys():
            appearance_count_frequency_dict[i] = 1
        else:
            appearance_count_frequency_dict[i] += 1
    return appearance_count_frequency_dict


def draw_log_log_graphs(appearance_counts_list):
    independent_ref_ids = range(0, len(appearance_counts_list))

    appearance_counts_list.sort(reverse=True)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.loglog(independent_ref_ids, appearance_counts_list, linestyle='None', marker='.')
    plt.title(f"{dataset_name} - Loglog graph of citations appearance count frequencies")
    plt.xlabel("Each citation corresponds to a point")
    plt.ylabel("Appearance counts for each citation")
    plt.show()

    plt.scatter(independent_ref_ids, appearance_counts_list)
    plt.title(f"{dataset_name} - Scatter graph of citations appearance count frequencies")
    plt.xlabel("Each citation corresponds to a point")
    plt.ylabel("Appearance counts for each citation")
    plt.show()

    plt.semilogy(independent_ref_ids, appearance_counts_list, linestyle='None', marker='.')
    plt.title(f"{dataset_name} - Semi_log_y graph of citations appearance count frequencies")
    plt.xlabel(f"Each citation corresponds to a point | from no 1 to no {len(appearance_counts_list)}")
    plt.ylabel("Appearance counts for each citation")
    plt.show()


def write_out_statistic_files(appearance_count_dict, appearance_counts_list, appearance_count_frequency_dict):
    my_keys = list(appearance_count_dict.keys())
    my_keys.sort()
    sorted_appearance_count_dict = {i: appearance_count_dict[i] for i in my_keys}
    with open(f"../dataset_statistic_files/{dataset_name}/appearance_count_per_cit_name_dict.json", "w") as outfile:
        json.dump(sorted_appearance_count_dict, outfile)

    appearance_counts_list.sort(reverse=True)
    with open(f"../dataset_statistic_files/{dataset_name}/appearance_counts_list.json", "w") as outfile:
        json.dump(appearance_counts_list, outfile)

    my_keys = list(appearance_count_frequency_dict.keys())
    my_keys.sort(reverse=True)
    sorted_appearance_count_frequency_dict = {i: appearance_count_frequency_dict[i] for i in my_keys}
    with open(f"../dataset_statistic_files/{dataset_name}/appearance_count_frequency_dict.json", "w") as outfile:
        json.dump(sorted_appearance_count_frequency_dict, outfile)


appearance_counts = count_ref_appearances()
# find_min_and_max_ref_appearances(appearance_counts)

all_appearance_counts = find_all_appearance_counts(appearance_counts)
# draw_histogram_of_all_cit_per_context(appearance_counts)
# draw_histogram_of_partial_cit_per_context(appearance_counts)

# find_how_many_cites_have_more_than_75_appearances_and_less_than_7(appearance_counts)

_, _ = calculate_average_and_median_of_cit_per_ref(all_appearance_counts)

draw_log_log_graphs(all_appearance_counts)

appearance_count_frequencies = create_appearance_count_frequency_dict(all_appearance_counts)
write_out_statistic_files(appearance_counts, all_appearance_counts, appearance_count_frequencies)
