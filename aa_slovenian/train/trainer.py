import argparse

import evaluate
import huggingface_hub
from datasets import load_dataset
from sklearn.metrics import precision_recall_fscore_support
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer, \
    TrainingArguments

huggingface_hub.login(token="hf_PUT_YOUR_TOKEN_HERE", add_to_git_credential=True)
huggingface_username = "gregorgabrovsek"

datasets = {
    "5_WithOOC": (f"{huggingface_username}/RTVCommentsTop5UsersWithOOC", f"{huggingface_username}/SloBertAA_Top5_WithOOC_082023"),
    "5_WithoutOOC": (f"{huggingface_username}/RTVCommentsTop5UsersWithoutOOC", f"{huggingface_username}/SloBertAA_Top5_WithoutOOC_082023"),
    "10_WithOOC": (f"{huggingface_username}/RTVCommentsTop10UsersWithOOC", f"{huggingface_username}/SloBertAA_Top10_WithOOC_082023"),
    "10_WithoutOOC": (f"{huggingface_username}/RTVCommentsTop10UsersWithoutOOC", f"{huggingface_username}/SloBertAA_Top10_WithoutOOC_082023"),
    "20_WithOOC": (f"{huggingface_username}/RTVCommentsTop20UsersWithOOC", f"{huggingface_username}/SloBertAA_Top20_WithOOC_082023"),
    "20_WithoutOOC": (f"{huggingface_username}/RTVCommentsTop20UsersWithoutOOC", f"{huggingface_username}/SloBertAA_Top20_WithoutOOC_08202"),
    "50_WithOOC": (f"{huggingface_username}/RTVCommentsTop50UsersWithOOC", f"{huggingface_username}/SloBertAA_Top50_WithOOC_082023"),
    "50_WithoutOOC": (f"{huggingface_username}/RTVCommentsTop50UsersWithoutOOC", f"{huggingface_username}/SloBertAA_Top50_WithoutOOC_082023"),
    "100_WithOOC": (f"{huggingface_username}/RTVCommentsTop100UsersWithOOC", f"{huggingface_username}/SloBertAA_Top100_WithOOC_082023"),
    "100_WithoutOOC": (f"{huggingface_username}/RTVCommentsTop100UsersWithoutOOC", f"{huggingface_username}/SloBertAA_Top100_WithoutOOC_082023"),

    "5_WithoutOOC_IMDB": (f"{huggingface_username}/imdb1m-top-5-users", f"{huggingface_username}/BERfT_AA_IMDB_Top5_WithoutOOC_082023"),
    "10_WithoutOOC_IMDB": (f"{huggingface_username}/imdb1m-top-10-users", f"{huggingface_username}/BERT_AA_IMDB_Top10_WithoutOOC_082023"),
    "25_WithoutOOC_IMDB": (f"{huggingface_username}/imdb1m-top-25-users", f"{huggingface_username}/BERT_AA_IMDB_Top25_WithoutOOC_082023"),
    "50_WithoutOOC_IMDB": (f"{huggingface_username}/imdb1m-top-50-users", f"{huggingface_username}/BERT_AA_IMDB_Top50_WithoutOOC_082023"),
    "100_WithoutOOC_IMDB": (f"{huggingface_username}/imdb1m-top-100-users", f"{huggingface_username}/BERT_AA_IMDB_Top100_WithoutOOC_082023"),
}

# Parse the command line arguments
parser = argparse.ArgumentParser()
# Dataset size, is required, no default
parser.add_argument("--dataset_size", type=int, help="Dataset size")
# Has OOC, is required, no default
parser.add_argument("--has_ooc", type=int, help="Has OOC")
# Use multilingual BERT, is required, no default
parser.add_argument("--use_multilingual_bert", type=int, help="Use multilingual BERT")
# Use IMDB dataset, is required, no default
parser.add_argument("--use_imdb", type=int, help="Use IMDB dataset")

args = parser.parse_args()
# Print all arguments
print(args)
# Make sure that if use_imdb is true, use_multilingual_bert is true
if args.use_imdb == 1 and (args.use_multilingual_bert == 0 or args.has_ooc == 1):
    raise ValueError("If use_imdb is true, use_multilingual_bert must be true as well and has_ooc must be false!")

# Set the dataset size
DATASET_SIZE = args.dataset_size
# Set the OOC flag
HAS_OOC = args.has_ooc == 1
# Set the multilingual BERT flag
USE_MULTILINGUAL_BERT = args.use_multilingual_bert == 1
# Construct the SELECTED string
SELECTED = str(DATASET_SIZE) + "_" + ("WithOOC" if HAS_OOC else "WithoutOOC") + ("_IMDB" if args.use_imdb == 1 else "")
print("SELECTED: " + SELECTED)

rtv_comments = load_dataset(datasets[SELECTED][0])

model_base = "bert-base-multilingual-uncased" if USE_MULTILINGUAL_BERT else "EMBEDDIA/sloberta"
tokenizer = AutoTokenizer.from_pretrained(model_base)
print(model_base)


def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)


tokenized_comments = rtv_comments.map(preprocess_function, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

accuracy = evaluate.load("accuracy")


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
    acc = accuracy.compute(references=labels, predictions=preds)
    return {
        'accuracy': acc["accuracy"],
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


id2label = {
    i: str(i)
    for i in range(DATASET_SIZE)
}
label2id = {
    str(i): i
    for i in range(DATASET_SIZE)
}
if HAS_OOC:
    id2label[DATASET_SIZE] = "OOC"
    label2id["OOC"] = DATASET_SIZE

print(id2label)
print(label2id)

n_o_l = DATASET_SIZE
if HAS_OOC:
    n_o_l += 1
model = AutoModelForSequenceClassification.from_pretrained(
    model_base, num_labels=n_o_l, id2label=id2label, label2id=label2id
)

output_dir = datasets[SELECTED][1]
if USE_MULTILINGUAL_BERT:
    output_dir += "_MultilingualBertBase"
training_args = TrainingArguments(
    output_dir=output_dir,
    learning_rate=2e-5,
    per_device_train_batch_size=12,
    per_device_eval_batch_size=12,
    num_train_epochs=10,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,
    load_best_model_at_end=True,
    push_to_hub=True,
    report_to=["tensorboard"],
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_comments["train"],
    eval_dataset=tokenized_comments["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.push_to_hub()
