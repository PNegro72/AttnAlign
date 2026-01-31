from metric_2 import attn_align_score
import json

if __name__ == "__main__":
    model1 = "microsoft/deberta-v3-base"  # recomendado con 32GB RAM
    model2 = "xlm-roberta-base"
    model3 = "Qwen/Qwen2.5-1.5B-Instruct"
    # model = "xlm-roberta-base"  # luego probá: "Qwen/Qwen2.5-1.5B-Instruct"

    good, bad = [], []
    with open("Data/qa_eval_dataset.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            sample = json.loads(line)
            r = attn_align_score(sample["question"], sample["answer"], model2)
            print(sample["id"], r["AttnAlign"], r)
            if "correct" in sample["id"]:
                good.append(r["AttnAlign"])
            else:
                bad.append(r["AttnAlign"])

    print("Avg correct:", sum(good)/len(good))
    print("Avg incorrect:", sum(bad)/len(bad))            