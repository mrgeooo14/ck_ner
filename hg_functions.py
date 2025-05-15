import evaluate
import numpy as np


### For NER
metric = evaluate.load("seqeval")


def get_label_mappings(dataset):
    """
    Extract label list and create id mappings from a dataset with 'ner_tags'.
    """
    label_list = sorted({tag for example in dataset["train"] for tag in example["ner_tags"]})
    label_to_id = {label: i for i, label in enumerate(label_list)}
    id_to_label = {i: label for label, i in label_to_id.items()}
    return label_list, label_to_id, id_to_label


def tokenize_and_align_labels_for_ner(example, tokenizer, label_to_id):
    ### get all labels and convert them to ids required be the model    
    tokenized = tokenizer(
        example["tokens"],
        is_split_into_words=True, ##we have it already split from spacy
        truncation=True,
        padding="max_length",
        max_length=128
    )
    word_ids = tokenized.word_ids()
    ### adapt labels to subwords
    labels = []
    prev_word_id = None

    for word_id in word_ids:
        if word_id is None:
            labels.append(-100)
        elif word_id != prev_word_id:
            labels.append(label_to_id[example["ner_tags"][word_id]])
        else:
            labels.append(-100)
        prev_word_id = word_id

    tokenized["labels"] = labels
    return tokenized



def compute_metrics(p, id_to_label):
    """
    Computes seqeval metrics (precision, recall, f1, accuracy) 
    for token classification predictions.
    """
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [id_to_label[p] for (p, l) in zip(pred, lab) if l != -100]
        for pred, lab in zip(predictions, labels)
    ]
    true_labels = [
        [id_to_label[l] for (p, l) in zip(pred, lab) if l != -100]
        for pred, lab in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)

    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }