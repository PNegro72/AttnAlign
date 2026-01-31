import re, math, torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

# -------------------------------
# Stopwords ES / EN
# -------------------------------
SPANISH_STOP = {
    "el","la","los","las","un","una","unos","unas","de","del","al","a","ante","bajo","cabe","con","contra",
    "desde","durante","en","entre","hacia","hasta","mediante","para","por","según","sin","so","sobre","tras",
    "y","o","u","e","que","como","cuál","cual","cuáles","cuales","cuándo","cuando","dónde","donde","qué","quién","quien",
    "es","son","ser","se","su","sus","tu","tus","mi","mis"
}
EN_STOP = {"the","a","an","of","to","in","on","for","with","and","or","is","are","be","as","at","by","from","that","this"}

# -------------------------------
# Utils
# -------------------------------
def _clean_text(t: str) -> str:
    t = re.sub(r"<[^>]+>", " ", t)                 # HTML/SSML
    t = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", t) # markdown links
    t = re.sub(r"[*_`#>]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def _mask_stopwords(tokenizer, ids):
    toks = tokenizer.convert_ids_to_tokens(ids.tolist())
    keep = []
    for tok in toks:
        low = tok.lower().strip("▁Ġ·.,;:!?¿¡()[]{}\"'«»…")
        if (not low) or (low in SPANISH_STOP) or (low in EN_STOP):
            keep.append(False)
        else:
            keep.append(True)
    return torch.tensor(keep, dtype=torch.bool)

# -------------------------------
# Main metric
# -------------------------------
@torch.inference_mode()
def attn_align_score(Q: str, A: str, model_name: str,
                     last_L: int = 4, weights=(0.35,0.5,0.15),
                     length_norm: float = 0.15, device="auto"):
    """
    Métrica AttnAlign universal para modelos encoder o decoder.
    - Q, A: strings
    - model_name: HF repo id (ej: 'dccuchile/bert-base-spanish-wwm-uncased' o 'Qwen2.5-1.5B')
    """
    Q = _clean_text(Q); A = _clean_text(A)

    # Tokenizer
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    # Intentar cargar como decoder primero
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        mdl = AutoModelForCausalLM.from_pretrained(
            model_name,
            output_attentions=True,
            output_hidden_states=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        mdl = mdl.to(device)
        is_decoder = bool(getattr(mdl.config, "is_decoder", False))
    except Exception:
        mdl = AutoModel.from_pretrained(
            model_name,
            output_attentions=True,
            output_hidden_states=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        mdl = mdl.to(device)
        is_decoder = False


    # ----------------------------
    # DECODER-ONLY PATH
    # ----------------------------
    if is_decoder:
        bos = tok.bos_token or tok.eos_token or ""
        sep = tok.sep_token or tok.eos_token or tok.eos_token
        prefix = (bos + Q + " " + sep).strip()
        text = (prefix + " " + A).strip()
        enc = tok(text, return_tensors="pt").to(mdl.device)
        out = mdl(**enc, output_attentions=True, output_hidden_states=True, use_cache=False)
        input_ids = enc.input_ids[0]; T = input_ids.size(0)

        atts = out.attentions[-last_L:]
        Wbar = torch.stack([a[0].mean(0) for a in atts]).mean(0)  # (T,T)
        hs_last = out.hidden_states[-1][0]

        q_ids = tok(prefix, return_tensors="pt").input_ids[0]
        s = q_ids.size(0)
        Q_idx = torch.arange(0, s-1) if s-1 > 0 else torch.tensor([])
        A_idx = torch.arange(s, T) if T > s else torch.tensor([])

    # ----------------------------
    # ENCODER PATH
    # ----------------------------
    else:
        sep = tok.sep_token or "[SEP]"
        if tok.sep_token is None and "[SEP]" not in tok.get_vocab():
            sep = "."
        text = (Q + " " + sep + " " + A).strip()
        enc = tok(text, return_tensors="pt", add_special_tokens=True).to(mdl.device)
        out = mdl(**enc, output_attentions=True, output_hidden_states=True)
        input_ids = enc.input_ids[0]; T = input_ids.size(0)

        atts = out.attentions[-last_L:]
        Wbar = torch.stack([a.mean(1) for a in atts]).mean(0)[0]  # (T,T)
        hs_last = out.hidden_states[-1][0]

        if tok.sep_token_id is not None:
            sep_positions = (input_ids == tok.sep_token_id).nonzero(as_tuple=False).flatten()
            split = int(sep_positions[0].item()) if len(sep_positions) > 0 else None
        else:
            split = None
        if split is None:
            q_ids = tok(Q, return_tensors="pt", add_special_tokens=True).input_ids[0]
            split = q_ids.size(0) - 1
        Q_idx = torch.arange(0, split) if split and split > 0 else torch.tensor([])
        A_idx = torch.arange(split+1, T) if split and (split+1) < T else torch.tensor([])

    if len(A_idx) == 0 or len(Q_idx) == 0:
        return {"QAtr": 0.0, "AliSim": 0.0, "Focus": 0.0,
                "LenRatio": 0.0, "LenPenalty": 1.0, "AttnAlign": 0.0,
                "is_decoder": is_decoder, "model_used": model_name}

    # ----------------------------
    # Procesar pregunta
    # ----------------------------
    keep_Q = _mask_stopwords(tok, input_ids[Q_idx])
    Q_masked = Q_idx[keep_Q] if keep_Q.any() else Q_idx
    if len(Q_masked) == 0: Q_masked = Q_idx

    # IDF heurístico
    q_toks = tok.convert_ids_to_tokens(input_ids[Q_masked].tolist())
    idf = torch.tensor([max(1.0, 6.0 - len(re.sub(r'[^A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9]', '', t)) )
                        for t in q_toks], dtype=torch.float32)
    idf = idf / idf.max()

    AtoQ_full = Wbar[A_idx][:, Q_masked] * idf

    # k dinámico
    # k = max(5, min(12, int(0.3 * AtoQ_full.size(1))))
    k = min(max(5, int(0.3 * AtoQ_full.size(1))), AtoQ_full.size(1))
    if k == 0:
        return {"QAtr": 0.0, "AliSim": 0.0, "Focus": 0.0,
                "LenRatio": 0.0, "LenPenalty": 1.0, "AttnAlign": 0.0,
                "is_decoder": is_decoder, "model_used": model_name}

    topk_vals, topk_idx = torch.topk(AtoQ_full, k, dim=1)
    AtoQ = torch.zeros_like(AtoQ_full).scatter(1, topk_idx, topk_vals)

    # QAtr
    row_sum = AtoQ.sum(dim=1, keepdim=True) + 1e-12
    QAtr = row_sum.mean().item()

    # AliSim
    alphas = AtoQ / row_sum
    H_Q = hs_last[Q_masked]
    H_A = hs_last[A_idx]
    q_tilde = alphas @ H_Q
    AliSim = F.cosine_similarity(H_A, q_tilde, dim=1).mean().clamp(-1,1).add(1).div(2).item()

    # Focus
    ent = -(alphas.clamp_min(1e-12).log() * alphas).sum(dim=1)
    Focus = (1.0 - (ent / (math.log(k)+1e-12))).mean().clamp(0,1).item()

    # Longitud
    len_ratio = (len(A_idx) / max(1.0, float(len(Q_idx))))
    length_penalty = 1.0 / (1.0 + length_norm * max(0.0, len_ratio - 2.0))

    w1, w2, w3 = weights
    base = (w1*QAtr + w2*AliSim + w3*Focus)
    score = 100.0 * (base * length_penalty)

    return {
        "QAtr": float(QAtr),
        "AliSim": float(AliSim),
        "Focus": float(Focus),
        "LenRatio": float(len_ratio),
        "LenPenalty": float(length_penalty),
        "AttnAlign": float(score),
        "is_decoder": is_decoder,
        "model_used": model_name
    }

# -------------------------------
# Ejemplo rápido
# -------------------------------
if __name__ == "__main__":
    model1 = "microsoft/deberta-v3-base"  # recomendado con 32GB RAM
    model2 = "xlm-roberta-base"
    model3 = "Qwen/Qwen2.5-1.5B-Instruct"
    Q = "¿Cuál es la capital de Francia?"
    A = "La capital de Francia es París."
    # print(attn_align_score(Q, A, model1))
    # print(attn_align_score(Q, A, model2))
    # print(attn_align_score(Q, A, model3))

    import json
    with open("Data/qa_eval_dataset.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            sample = json.loads(line)
            score = attn_align_score(sample["question"], sample["answer"], model2)
            print(sample["id"], score["AttnAlign"])