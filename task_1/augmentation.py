# %% [markdown]
# Doel: Verschillende datasets adhv prompts en augmentation in HuggingFace stijl.

# %% [markdown]
# # Load training data

# %%
import os
import pandas as pd

data_dir = os.path.join('task_1/dataset')
arguments_training_filepath = os.path.join(data_dir, 'arguments-training.tsv')
labels_training_filepath = os.path.join(data_dir, 'labels-training.tsv')
arguments_training = pd.read_csv(arguments_training_filepath, encoding='utf-8', sep='\t', header=0)
labels_training = pd.read_csv(labels_training_filepath, encoding='utf-8', sep='\t', header=0)


# %%
label_occurences = {}

for label in labels_training:
    if label != "Argument ID":
        number_of_occurences = labels_training[label].value_counts()[1]
        label_occurences[label] = number_of_occurences
        print(f"Number of {label}: {number_of_occurences}")

# %% [markdown]
# Three methods, so to input each sentence into all methods once and create class balance, you need three times the largest class number  (3*1992=5976) of instances per class.

# %%
NUMBER_OF_CLASS_INSTANCES = 5976

required_label_occurences = {}
for item in label_occurences:
    required_label_occurences[item] = NUMBER_OF_CLASS_INSTANCES - label_occurences[item]

# %% [markdown]
# # Paraphrasing

# %% [markdown]
# ### LM based

# %%
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("prithivida/parrot_paraphraser_on_T5")  
model = AutoModelForSeq2SeqLM.from_pretrained("prithivida/parrot_paraphraser_on_T5")

# %%
import random

print("LM")

complete_dataframe = pd.DataFrame()

# create a sample of a third of the required instances for augmentation
for label in required_label_occurences:
    print(label)
    to_create = round(required_label_occurences[label]/3)
    index_list = labels_training.index[labels_training[label] == 1].to_list()
    subset_df = arguments_training.loc[index_list]
    extended_df_ML = subset_df.sample(n=to_create, replace=True)

    # print(extended_df_ML)
    # LM based
    for _, row in extended_df_ML.iterrows():
        full_text = "The conclusion " + row['Conclusion'] + " " + row["Stance"] + " is against premise:" + row["Premise"]
        conclusion = "paraphrase: " + row['Conclusion']
        conclusion_encoding = tokenizer.encode_plus(conclusion,padding=True, return_tensors="pt")
        conclusion_input_ids, conclusion_attention_masks = conclusion_encoding["input_ids"], conclusion_encoding["attention_mask"]
        conclusion_outputs = model.generate(
            input_ids=conclusion_input_ids, attention_mask=conclusion_attention_masks,
            max_length=256,
            do_sample=True,
            top_k=120,
            top_p=0.95,
            early_stopping=True,
            num_return_sequences=5
        )
        paraphrased_conclusion = tokenizer.decode(conclusion_outputs[random.randint(0, 4)], skip_special_tokens=True,clean_up_tokenization_spaces=True)

        premise = "paraphrase: " + row['Premise'] 
        premise_encoding = tokenizer.encode_plus(premise,padding=True, return_tensors="pt")
        premise_input_ids, premise_attention_masks = premise_encoding["input_ids"], premise_encoding["attention_mask"]

        premise_outputs = model.generate(
            input_ids=premise_input_ids, attention_mask=premise_attention_masks,
            max_length=256,
            do_sample=True,
            top_k=120,
            top_p=0.9,
            early_stopping=True,
            num_return_sequences=5
        )
        paraphrased_premise = tokenizer.decode(premise_outputs[random.randint(0, 4)], skip_special_tokens=True,clean_up_tokenization_spaces=True)
        
        row['Conclusion'] = paraphrased_conclusion
        row['Premise'] = paraphrased_premise
        print("The conclusion " + row['Conclusion'] + " " + row["Stance"] + " is against premise:" + row["Premise"])

    # merge augmented and normal df together
    complete_dataframe = pd.concat([complete_dataframe, extended_df_ML])

complete_dataframe.to_csv("LM.csv")

# %% [markdown]
# ### Thesaurus based

# %%
from textattack.augmentation import WordNetAugmenter

print("thesaurus")

wordnet_aug = WordNetAugmenter(pct_words_to_swap=0.3, high_yield=False)

complete_dataframe = pd.DataFrame()

for label in required_label_occurences:
    print(label)
    to_create = round(required_label_occurences[label]/3)
    index_list = labels_training.index[labels_training[label] == 1].to_list()
    subset_df = arguments_training.loc[index_list]
    extended_df_TH = subset_df.sample(n=to_create, replace=True)

    for _, row in extended_df_TH.iterrows():
        full_text = "The conclusion " + row['Conclusion'] + " " + row["Stance"] + " is against premise:" + row["Premise"]
        # augmented_full = wordnet_aug.augment(full_text)
        augmented_premise = wordnet_aug.augment(row["Premise"])
        augmented_conclusion = wordnet_aug.augment(row['Conclusion'])
        # print(full_text)
        # print(augmented_full)
        row['Conclusion'] = augmented_conclusion[0]
        row['Premise'] = augmented_premise[0]
        augmented_text = "The conclusion " + row['Conclusion'] + " " + row["Stance"] + " is against premise:" + row["Premise"]
        # print(augmented_text)

    # merge augmented and normal df together
    complete_dataframe = pd.concat([complete_dataframe, extended_df_ML])

complete_dataframe.to_csv("thesaurus.csv")

# %% [markdown]
# # Noising

# %%
from textattack.augmentation import EasyDataAugmenter

print("noise")
complete_dataframe = pd.DataFrame()

eda_aug = EasyDataAugmenter(pct_words_to_swap=0.2)

for label in required_label_occurences:
    print("label")
    to_create = round(required_label_occurences[label]/3)
    index_list = labels_training.index[labels_training[label] == 1].to_list()
    subset_df = arguments_training.loc[index_list]
    extended_df_NO = subset_df.sample(n=to_create, replace=True)

    for _, row in extended_df_NO.iterrows():
        full_text = "The conclusion " + row['Conclusion'] + " " + row["Stance"] + " is against premise:" + row["Premise"]
        augmented_premise = eda_aug.augment(row["Premise"])
        augmented_conclusion = eda_aug.augment(row['Conclusion'])
        # print(full_text)
        row['Conclusion'] = augmented_conclusion[0]
        row['Premise'] = augmented_premise[0]
        augmented_text = "The conclusion " + row['Conclusion'] + " " + row["Stance"] + " is against premise:" + row["Premise"]
        # print(augmented_text)

    # merge augmented and normal df together
    complete_dataframe = pd.concat([complete_dataframe, extended_df_ML])

complete_dataframe.to_csv("noise.csv")


