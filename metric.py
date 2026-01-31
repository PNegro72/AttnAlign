import torch, math
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

@torch.inference_mode()
def attn_align_score(question: str, answer: str, model, tokenizer, last_L: int = 4, weights=(0.4,0.4,0.2)):
    # 1) Build sequence: [BOS] Q [SEP] A
    bos = tokenizer.bos_token or tokenizer.eos_token or ""
    sep = tokenizer.sep_token or tokenizer.eos_token or tokenizer.eos_token
    text = (bos + question + " " + sep + " " + answer).strip()
    enc = tokenizer(text, return_tensors="pt")
    T = enc.input_ids.size(1)

    out = model(**enc, output_attentions=True, output_hidden_states=True, use_cache=False)
    hs_last = out.hidden_states[-1][0]                       # (T, d)
    atts = out.attentions[-last_L:]                          # list last_L layers, each (1, H, T, T)

    # 2) Aggregate heads & layers → Wbar in (T, T)
    Wbar = torch.stack([a[0].mean(0) for a in atts]).mean(0) # mean over heads, then layers

    # 3) Locate spans: Q and A
    # Heurística: índice del primer token del answer ≈ después del sep
    with tokenizer.as_target_tokenizer():  # no-op for causal; kept explicit
        q_ids = tokenizer((bos + question + " " + sep).strip(), return_tensors="pt").input_ids[0]
    s = q_ids.size(0)                       # cutoff (A empieza en s)
    Q_idx = torch.arange(0, s-1)            # [BOS..SEP-1] = pregunta
    A_idx = torch.arange(s, T)              # respuesta

    if len(A_idx) == 0 or len(Q_idx) == 0:  # safety
        return {"QAtr": 0.0, "AliSim": 0.0, "Focus": 0.0, "AttnAlign": 0.0}

    # 4) QAtr: atención de cada token de A hacia tokens de Q
    AtoQ = Wbar[A_idx][:, Q_idx]            # (|A|, |Q|)
    QAtr = AtoQ.sum(dim=1).mean().item()    # promedio en tokens de A → [0,1]

    # 5) AliSim: cos(H_t, sum_j alpha_{t,j} * H_j) con alphas normalizados en Q
    alphas = AtoQ / (AtoQ.sum(dim=1, keepdim=True) + 1e-12)  # (|A|, |Q|)
    H_Q = hs_last[Q_idx]                    # (|Q|, d)
    H_A = hs_last[A_idx]                    # (|A|, d)
    q_tilde = alphas @ H_Q                  # (|A|, d)
    AliSim = F.cosine_similarity(H_A, q_tilde, dim=1).mean().clamp(-1,1).add(1).div(2).item()  # map [-1,1]→[0,1]

    # 6) Focus: 1 - H(alpha)/log(|Q|)
    ent = -(alphas.clamp_min(1e-12).log() * alphas).sum(dim=1)        # (|A|)
    Focus = (1.0 - (ent / math.log(len(Q_idx)))).mean().clamp(0,1).item()

    # 7) Weighted score → [0..100]
    w1, w2, w3 = weights
    score = 100.0 * (w1*QAtr + w2*AliSim + w3*Focus)
    return {"QAtr": float(QAtr), "AliSim": float(AliSim), "Focus": float(Focus), "AttnAlign": float(score)}

# ---------- Ejemplo de uso ----------
if __name__ == "__main__":
    name = "sshleifer/tiny-gpt2"  # o "gpt2" / "TinyLlama/TinyLlama-1.1B-Chat-v1.0" (más pesado)
    tok = AutoTokenizer.from_pretrained(name)
    mdl = AutoModelForCausalLM.from_pretrained(name)
    Q = "Cómo hago para que depositen el sueldo en este banco"
    A = "¡Qué bueno que confíes en nosotros!\\nPara **abrir tu cuenta sueldo** ingresá tus datos en [Sueldos <phoneme alphabet=\"ipa\" ph= \"ɑ.ɪzibiziˈ\">ICBC](https://www.sueldos.icbc.com.ar)</phoneme> desde nuestra web y nos comunicaremos con vos dentro de las próximas 48<sub alias=\"horas\">h</sub> hábiles.\\nTambién podés acercarte a cualquiera de nuestras sucursales y solicitar la apertura de tu cuenta, solo tenés que presentar tu recibo de haberes.\\nSi ya tenés una caja de ahorro en pesos con nosotros no te olvides de informarle a tu empleador tu número de cuenta y CBU."
    print(attn_align_score(Q, A, mdl, tok))

