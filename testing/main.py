import json
from sklearn.metrics import f1_score, accuracy_score

def query(context):
    #TODO: Fill with actual route/rag
    return ("provable", "True")

def main():
    with open("test-dataset.json", "r", encoding="utf-8") as f:
        data = json.load(f)["data"]

    true_query_types = []
    pred_query_types = []
    true_truth_values = []
    pred_truth_values = []

    for example in data:
        context = example["context"]
        true_query_type = example["query_type"]
        true_truth_value = str(example["truth_value"])

        pred_query_type, pred_truth_value = query(context)

        true_query_types.append(true_query_type)
        pred_query_types.append(pred_query_type)
        true_truth_values.append(true_truth_value)
        pred_truth_values.append(pred_truth_value)

    query_type_f1 = f1_score(true_query_types, pred_query_types, average='macro')
    query_type_acc = accuracy_score(true_query_types, pred_query_types)

    truth_value_f1 = f1_score(true_truth_values, pred_truth_values, average='macro')
    truth_value_acc = accuracy_score(true_truth_values, pred_truth_values)

    print("Query Type - F1 Score:", query_type_f1)
    print("Query Type - Accuracy:", query_type_acc)
    print("Truth Value - F1 Score:", truth_value_f1)
    print("Truth Value - Accuracy:", truth_value_acc)


if __name__ == "__main__":
    main()
    