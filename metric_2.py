import re, math, torch, json
import torch.nn.functional as F
from functools import lru_cache
from transformers import AutoConfig, AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoModelForSequenceClassification
from transformers.utils import logging as hf_logging

# Silenciar warnings molestos
hf_logging.set_verbosity_error()

SPANISH_STOP = {
    "el","la","los","las","un","una","unos","unas","de","del","al","a","ante","bajo","cabe","con","contra",
    "desde","durante","en","entre","hacia","hasta","mediante","para","por","según","sin","so","sobre","tras",
    "y","o","u","e","que","como","cuál","cual","cuáles","cuales","cuándo","cuando","dónde","donde","qué","quién","quien",
    "es","son","ser","se","su","sus","tu","tus","mi","mis"
}
EN_STOP = {"the","a","an","of","to","in","on","for","with","and","or","is","are","be","as","at","by","from","that","this"}

def _clean_text(t: str) -> str:
    t = re.sub(r"<[^>]+>", " ", t)
    t = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", t)  # [texto](link)
    t = re.sub(r"[*_`#>]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def _mask_stopwords(tokenizer, ids):
    toks = tokenizer.convert_ids_to_tokens(ids.tolist())
    keep = []
    for tok in toks:
        low = tok.lower().strip("▁Ġ·.,;:!?¿¡()[]{}\"'«»…")
        keep.append(bool(low) and (low not in SPANISH_STOP) and (low not in EN_STOP))
    return torch.tensor(keep, dtype=torch.bool)

@lru_cache(maxsize=1)
def load_nli():
    nli_name = "joeddav/xlm-roberta-large-xnli"
    nli_tok = AutoTokenizer.from_pretrained(nli_name, use_fast=True)
    nli_mdl = AutoModelForSequenceClassification.from_pretrained(nli_name).to("cuda" if torch.cuda.is_available() else "cpu")
    nli_mdl.eval()
    return nli_tok, nli_mdl

def lexical_hit_rate(q_str: str, a_str: str):
    q = _clean_text(q_str).lower()
    a = _clean_text(a_str).lower()
    q_toks = re.findall(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9]+", q)
    a_set  = set(re.findall(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9]+", a))
    q_toks = [t for t in q_toks if t not in SPANISH_STOP and t not in EN_STOP and len(t) > 2]
    if not q_toks:
        return 0.0
    hits = sum(1 for t in q_toks if t in a_set)
    return hits / len(q_toks)

@lru_cache(maxsize=4)
def load_arbiter(model_name: str):
    """Carga modelo y tokenizer UNA VEZ y devuelve (tok, mdl, is_decoder)."""
    cfg = AutoConfig.from_pretrained(model_name)
    if getattr(cfg, "is_encoder_decoder", False):
        mdl = AutoModel.from_pretrained(model_name, output_attentions=True, output_hidden_states=True)
        is_decoder = False
    elif getattr(cfg, "is_decoder", False) or (getattr(cfg, "model_type", "") in {"qwen2","qwen2_moe","qwen"}):
        mdl = AutoModelForCausalLM.from_pretrained(model_name, output_attentions=True, output_hidden_states=True)
        is_decoder = True
    else:
        mdl = AutoModel.from_pretrained(model_name, output_attentions=True, output_hidden_states=True)
        is_decoder = False

    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token if tok.eos_token is not None else tok.unk_token

    device = "cuda" if torch.cuda.is_available() else "cpu"
    mdl = mdl.to(device)
    return tok, mdl, is_decoder

@torch.inference_mode()
def attn_align_score(Q: str, A: str, model_name: str,
                     last_L: int = 4, weights=(0.5,0.2,0.3),
                     length_norm: float = 0.3):
    """
    Métrica AttnAlign: QAtr (masa A->Q), AliSim (consistencia semántica), Focus (nitidez).
    Soporta encoders (p.ej. xlm-roberta-base) y decoders (p.ej. Qwen/Qwen2.5-1.5B-Instruct).
    """
    Q = _clean_text(Q); A = _clean_text(A)
    tok, mdl, is_decoder = load_arbiter(model_name)

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
        Q_idx = torch.arange(0, s-1, device=mdl.device, dtype=torch.long) if s-1 > 0 else torch.tensor([], device=mdl.device, dtype=torch.long)
        A_idx = torch.arange(s, T,   device=mdl.device, dtype=torch.long) if T > s   else torch.tensor([], device=mdl.device, dtype=torch.long)
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
            pos = (input_ids == tok.sep_token_id).nonzero(as_tuple=False).flatten()
            split = int(pos[0].item()) if len(pos) > 0 else None
        else:
            split = None
        if split is None:
            q_ids2 = tok(Q, return_tensors="pt", add_special_tokens=True).input_ids[0]
            split = q_ids2.size(0) - 1
        Q_idx = torch.arange(0, split, device=mdl.device, dtype=torch.long) if (split is not None and split > 0) else torch.tensor([], device=mdl.device, dtype=torch.long)
        A_idx = torch.arange(split+1, T, device=mdl.device, dtype=torch.long) if (split is not None and (split+1) < T) else torch.tensor([], device=mdl.device, dtype=torch.long)

    if len(A_idx) == 0 or len(Q_idx) == 0:
        return {"QAtr": 0.0, "AliSim": 0.0, "Focus": 0.0, "LenRatio": 0.0, "LenPenalty": 1.0, "AttnAlign": 0.0,
                "is_decoder": is_decoder, "model_used": model_name}

    # Stopwords + IDF en Q
    keep_Q = _mask_stopwords(tok, input_ids[Q_idx].cpu())
    Q_masked = Q_idx[keep_Q.to(Q_idx.device)] if keep_Q.any() else Q_idx
    if len(Q_masked) == 0: Q_masked = Q_idx

    q_toks = tok.convert_ids_to_tokens(input_ids[Q_masked].tolist())
    idf_vals = [max(1.0, 6.0 - len(re.sub(r'[^A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9]', '', t))) for t in q_toks]
    idf = torch.tensor(idf_vals, dtype=torch.float32, device=Wbar.device)
    idf = idf / (idf.max() if idf.max() > 0 else 1.0)

    AtoQ_full = Wbar[A_idx][:, Q_masked] * idf  # (|A|, |Qm|)
    qcols = AtoQ_full.size(1)
    k = min(max(5, int(0.3 * qcols)), qcols)
    if k <= 0:
        return {"QAtr": 0.0, "AliSim": 0.0, "Focus": 0.0, "LenRatio": 0.0, "LenPenalty": 1.0, "AttnAlign": 0.0,
                "is_decoder": is_decoder, "model_used": model_name}

    topk_vals, topk_idx = torch.topk(AtoQ_full, k, dim=1)
    AtoQ = torch.zeros_like(AtoQ_full).scatter(1, topk_idx, topk_vals)

    # QAtr
    row_sum = AtoQ.sum(dim=1, keepdim=True) + 1e-12
    QAtr = float(row_sum.mean().item())

    # AliSim
    alphas = AtoQ / row_sum
    H_Q = hs_last[Q_masked]
    H_A = hs_last[A_idx]
    q_tilde = alphas @ H_Q
    AliSim = float(F.cosine_similarity(H_A, q_tilde, dim=1).mean().clamp(-1,1).add(1).div(2).item())

    # Focus (suavizado)
    ent = -(alphas.clamp_min(1e-12).log() * alphas).sum(dim=1)
    Focus = float((1.0 - (ent / (math.log(max(k, 2))+1e-12))).mean().clamp(0,1).item())

    # Longitud
    len_ratio = float(len(A_idx) / max(1.0, float(len(Q_idx))))
    length_penalty = 1.0 / (1.0 + length_norm * max(0.0, len_ratio - 2.0))

    # Gating de AliSim por atención
    w1, w2, w3 = weights  # QAtr, AliSim, Focus
    tau1, tau2 = 0.25, 0.25
    gate = 1/(1+math.exp(-4.0*(QAtr - tau1))) * 1/(1+math.exp(-4.0*(Focus - tau2)))

    ali_gated = AliSim * gate

    base = (w1*QAtr + w2*ali_gated + w3*Focus)

    # score base (antes de penalizaciones semánticas externas)
    score_base = 100.0 * (base * length_penalty)

    # --- NLI penalty (suave, con auto-mapeo de etiquetas) ---
    try:
        nli_tok, nli_mdl = load_nli()
        nli_enc = nli_tok(Q, A, return_tensors="pt", truncation=True, max_length=384).to(nli_mdl.device)
        nli_out = nli_mdl(**nli_enc)
        probs = F.softmax(nli_out.logits, dim=-1)[0]

        # Detectar índices por id2label (robusto a distintos modelos)
        id2label = getattr(nli_mdl.config, "id2label", None)
        if id2label:
            label_map = {v.lower(): int(k) for k, v in id2label.items()}
            idx_ent = label_map.get("entailment", 0)
            idx_con = label_map.get("contradiction", 2)
        else:
            # fallback razonable para XNLI
            idx_ent, idx_con = 0, 2

        p_entail = float(probs[idx_ent].item())
        p_contra = float(probs[idx_con].item())

        # afinidad NLI en [-1,1]
        aff = p_entail - p_contra
        aff01 = (aff + 1.0) / 2.0  # [0..1]

        # Mezcla suave: nunca baja de 0.75 (antes llegaba a 0.5)
        nli_factor = 0.75 + 0.25 * aff01  # [0.75..1.0]
    except Exception:
        nli_factor = 1.0


    # --- Lexical Hit Rate (suave + short-answer safe) ---
    LHR = lexical_hit_rate(Q, A)

    # contar tokens de respuesta (aprox por IDs o por regex si querés)
    try:
        a_len = len(A_idx)
    except Exception:
        a_len = len(re.findall(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9]+", _clean_text(A)))

    if a_len < 6 or len(A.split()) <= 2:
        # Respuesta muy corta: no castigamos
        lex_factor = 1.0
    else:
        # Piso 0.8; a lo sumo +0.2 por overlap
        lex_factor = 0.8 + 0.2 * max(0.0, min(1.0, LHR))


    # Score final
    score = float(score_base) * float(nli_factor) * float(lex_factor)

    return {
        "QAtr": QAtr,
        "AliSim": AliSim,
        "Focus": Focus,
        "LenRatio": len_ratio,
        "LenPenalty": float(length_penalty),
        "AttnAlign": float(score),
        "is_decoder": is_decoder,
        "model_used": model_name
    }

# ---------- Utils: best threshold con sklearn (fallback si no está) ----------
def best_threshold_sklearn(scores, labels):
    import numpy as np
    try:
        from sklearn.metrics import f1_score
    except ImportError:
        # Fallback mínimo si no hay sklearn: barrido por puntos únicos
        s = np.asarray(scores, dtype=float); y = np.asarray(labels, dtype=int)
        thr_candidates = np.unique(s)
        best_thr, best_f1 = 0.0, -1.0
        for t in thr_candidates:
            preds = (s >= t).astype(int)
            tp = int(((preds == 1) & (y == 1)).sum())
            fp = int(((preds == 1) & (y == 0)).sum())
            fn = int(((preds == 0) & (y == 1)).sum())
            prec = tp/(tp+fp) if (tp+fp)>0 else 0.0
            rec  = tp/(tp+fn) if (tp+fn)>0 else 0.0
            f1 = (2*prec*rec)/(prec+rec) if (prec+rec)>0 else 0.0
            if f1 > best_f1:
                best_thr, best_f1 = float(t), float(f1)
        return best_thr, best_f1

    # Camino con sklearn
    from sklearn.metrics import f1_score
    s = np.asarray(scores, dtype=float); y = np.asarray(labels, dtype=int)
    thr_candidates = np.unique(s)
    best_thr, best_f1 = 0.0, -1.0
    for t in thr_candidates:
        preds = (s >= t).astype(int)
        f1 = f1_score(y, preds)
        if f1 > best_f1:
            best_thr, best_f1 = float(t), float(f1)
    return best_thr, best_f1


# -------------------- Demo batch con thresholding --------------------
if __name__ == "__main__":
    # Elegí UNO (comenta el otro):
    model = "xlm-roberta-base"
    # model = "Qwen/Qwen2.5-1.5B-Instruct"

    weights = (0.55, 0.15, 0.30)   # QAtr, AliSim (gated), Focus
    length_norm = 0.30

    all_scores, all_labels = [], []
    good, bad = [], []

    with open("Data/qa_eval_dataset.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            s = json.loads(line)
            r = attn_align_score(s["question"], s["answer"], model, weights=weights, length_norm=length_norm)
            score = r["AttnAlign"]
            label = 1 if s["id"].endswith("-correct") else 0

            all_scores.append(score); all_labels.append(label)
            (good if label==1 else bad).append(score)

            print(s["question"])
            print(s["id"], round(score, 2), r)

    if good:
        print("Avg correct:", sum(good)/len(good))
    if bad:
        print("Avg incorrect:", sum(bad)/len(bad))

    # --- Buscar umbral óptimo por F1 ---
    thr, f1 = best_threshold_sklearn(all_scores, all_labels)
    preds = [1 if sc >= thr else 0 for sc in all_scores]

    # Métricas simples
    tp = sum((p==1 and y==1) for p,y in zip(preds, all_labels))
    fp = sum((p==1 and y==0) for p,y in zip(preds, all_labels))
    tn = sum((p==0 and y==0) for p,y in zip(preds, all_labels))
    fn = sum((p==0 and y==1) for p,y in zip(preds, all_labels))

    prec = tp/(tp+fp) if (tp+fp)>0 else 0.0
    rec  = tp/(tp+fn) if (tp+fn)>0 else 0.0
    acc  = (tp+tn)/len(all_labels) if all_labels else 0.0

    print("\n>>> Threshold óptimo (max F1): {:.4f}".format(thr))
    print("Accuracy={:.3f}  Precision={:.3f}  Recall={:.3f}  F1={:.3f}".format(acc, prec, rec, f1))
    print("Confusion Matrix: TP={}  FP={}  TN={}  FN={}".format(tp, fp, tn, fn))
