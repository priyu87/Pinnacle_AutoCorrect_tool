# ===========================
# Step 1: Install dependencies
# ===========================
!pip -q install numpy pandas regex wordfreq rapidfuzz transformers torch sentencepiece python-Levenshtein symspellpy gradio datasets --upgrade
!pip install pandas==2.2.2 numpy==1.26.4

# (Optional GPU acceleration + CV libs if needed)
# !pip install pandas==2.2.2 dask-cudf-cu12==25.6.0 cudf-cu12==25.6.0 opencv-python==4.12.0.88 opencv-contrib-python==4.12.0.88


# ===========================
# Step 2: Imports
# ===========================
import os, math, json, regex as re, string, itertools, random
from collections import Counter
import numpy as np
from wordfreq import top_n_list
from rapidfuzz.distance import Levenshtein
from symspellpy import SymSpell, Verbosity

import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from datasets import load_dataset
import gradio as gr


# ===========================
# Step 3: Seed and device
# ===========================
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===========================
# Step 4: Tokenizer utilities
# ===========================
WORD_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
USER_DICT = set()

def in_user_dict(w): return w.lower() in USER_DICT
def tokenize(text): return WORD_RE.findall(text)

def detokenize(tokens):
    out = []
    for i, t in enumerate(tokens):
        if i > 0 and re.match(r"\w", t) and re.match(r"\w", out[-1]):
            out.append(" " + t)
        elif i > 0 and t not in ",.!?:;%)]}" and out[-1] not in "([{" and not re.match(r"\s", t):
            out.append(" " + t)
        else:
            out.append(t)
    return "".join(out)

def is_word(t): return re.match(r"\w+\Z", t) is not None
def normalize_word(w): return w.lower()


# ===========================
# Step 5: Vocabulary
# ===========================
VOCAB_SIZE = 5000
top_words = top_n_list("en", n=VOCAB_SIZE)
vocab_freq = {w: 0 for w in top_words}


# ===========================
# Step 6: Dataset counts
# ===========================
unigrams, bigrams = Counter(), Counter()

def build_counts(lines_iter):
    U, B = Counter(), Counter()
    for line in lines_iter:
        toks = ["<s>"] + [t.lower() for t in tokenize(line)] + ["</s>"]
        U.update(toks)
        B.update(zip(toks[:-1], toks[1:]))
    return U, B

USE_WIKITEXT = True
if USE_WIKITEXT:
    try:
        ds = load_dataset("wikitext", "wikitext-2-raw-v1")
        lines = itertools.chain(ds["train"]["text"], ds["validation"]["text"], ds["test"]["text"])
        unigrams, bigrams = build_counts(lines)
    except Exception as e:
        print("Could not load WikiText-2:", e)


# ===========================
# Step 7: Keyboard distance
# ===========================
QWERTY_ROWS = [
    "`1234567890-",
    "qwertyuiop[]\\",
    "asdfghjkl;'",
    "zxcvbnm,./"
]

key_pos = {}
for r, row in enumerate(QWERTY_ROWS):
    for c, ch in enumerate(row):
        key_pos[ch] = (r, c)
        key_pos[ch.upper()] = (r, c)

def key_distance(a, b):
    if a == b: return 0.0
    pa, pb = key_pos.get(a), key_pos.get(b)
    if pa is None or pb is None: return 2.0
    return math.dist(pa, pb)

def edit_cost(src, dst):
    d = Levenshtein.distance(src, dst)
    bonus = 0.0
    for a, b in zip(src, dst):
        if a != b:
            bonus += max(0.0, 1.0 - min(1.5, key_distance(a, b))/1.5) * 0.2
    return max(0.0, d - bonus)


# ===========================
# Step 8: SymSpell
# ===========================
sym = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
for term, count in vocab_freq.items():
    sym.create_dictionary_entry(term, count)
print("SymSpell dictionary size:", len(sym.words))


# ===========================
# Step 9: Candidate generator
# ===========================
def edits1(word):
    alphabet = string.ascii_lowercase
    splits = [(word[:i], word[i:]) for i in range(len(word)+1)]
    deletes = [L+R[1:] for L,R in splits if R]
    transposes = [L+R[1]+R[0]+R[2:] for L,R in splits if len(R)>1]
    replaces = [L+c+R[1:] for L,R in splits if R for c in alphabet]
    inserts  = [L+c+R for L,R in splits for c in alphabet]
    return set(deletes + transposes + replaces + inserts)

def generate_candidates(word, topk=50):
    w = normalize_word(word)
    cand = set()
    for v in [Verbosity.TOP, Verbosity.CLOSEST, Verbosity.ALL]:
        for s in sym.lookup(w, v, max_edit_distance=2, transfer_casing=True):
            cand.add(s.term)
    E1 = edits1(w)
    E2 = set(e2 for e1 in list(E1)[:400] for e2 in edits1(e1))
    cand |= {e for e in (E1 | E2) if e in vocab_freq}
    cand = list(cand)
    cand.sort(key=lambda t: (-vocab_freq.get(t, 1), t))
    return cand[:topk]


# ===========================
# Step 10: Bigram scoring
# ===========================
def bigram_score(prev_tok, tok, next_tok):
    if not unigrams or not bigrams:
        return math.log1p(vocab_freq.get(tok.lower(), 1))
    p, t, n = prev_tok.lower(), tok.lower(), next_tok.lower()
    a = math.log1p(bigrams.get((p, t), 0) + 1)
    b = math.log1p(bigrams.get((t, n), 0) + 1)
    return a + b


# ===========================
# Step 11: Load MLM
# ===========================
MLM_NAME = "distilbert-base-uncased"
tok = AutoTokenizer.from_pretrained(MLM_NAME)
mlm = AutoModelForMaskedLM.from_pretrained(MLM_NAME).to(device).eval()


# ===========================
# Step 12: MLM logprob
# ===========================
@torch.no_grad()
def mlm_logprob(sentence_tokens, idx, candidate):
    tokens = sentence_tokens.copy()
    tokens[idx] = tok.mask_token
    text = detokenize(tokens)
    inputs = tok(text, return_tensors="pt").to(device)
    logits = mlm(**inputs).logits[0]
    mask_idx = (inputs["input_ids"][0] == tok.mask_token_id).nonzero().flatten()
    if len(mask_idx) == 0: return -1e9
    m = mask_idx[0].item()
    cand_ids = tok(candidate, add_special_tokens=False)["input_ids"]
    if not cand_ids: return -1e9
    log_probs = torch.log_softmax(logits[m], dim=-1)
    lp = 0.0
    for sub_id in cand_ids:
        lp += float(log_probs[sub_id].item())
    return lp / len(cand_ids)


# ===========================
# Step 13: Suspicious words
# ===========================
@torch.no_grad()
def looks_suspicious(tokens, i, thresh=-2.0):
    w = tokens[i]
    if not is_word(w) or in_user_dict(w):
        return False
    low = w.lower()
    if low not in vocab_freq:
        return True
    try:
        cur_lp = mlm_logprob(tokens, i, w)
        tmp = tokens.copy()
        tmp[i] = tok.mask_token
        text = detokenize(tmp)
        inputs = tok(text, return_tensors="pt").to(device)
        logits = mlm(**inputs).logits[0]
        m = (inputs["input_ids"][0] == tok.mask_token_id).nonzero().flatten()[0].item()
        topk_vals = torch.topk(torch.log_softmax(logits[m], dim=-1), k=5).values
        median_top5 = float(torch.median(topk_vals).item())
        return (cur_lp - median_top5) < thresh
    except Exception:
        return False


# ===========================
# Step 14: Split / Merge
# ===========================
@torch.no_grad()
def try_split_merge(tokens, i):
    suggestions = []
    w = tokens[i]
    if not is_word(w):
        return suggestions
    # Split
    for cut in range(1, len(w)):
        left, right = w[:cut], w[cut:]
        if left.lower() in vocab_freq and right.lower() in vocab_freq:
            cand_tokens = tokens[:i] + [left, right] + tokens[i+1:]
            lp = mlm_logprob(cand_tokens, i, left) + mlm_logprob(cand_tokens, i+1, right)
            suggestions.append(("split", f"{left} {right}", lp))
    # Merge
    if i+1 < len(tokens) and is_word(tokens[i+1]):
        merged = (w + tokens[i+1])
        if merged.lower() in vocab_freq:
            cand_tokens = tokens[:i] + [merged] + tokens[i+2:]
            lp = mlm_logprob(cand_tokens, i, merged)
            suggestions.append(("merge", merged, lp))
    return suggestions


# ===========================
# Step 15: Correction function
# ===========================
def preserve_case(src, dst):
    if src.istitle(): return dst.title()
    if src.isupper(): return dst.upper()
    if src.islower(): return dst.lower()
    return dst

def correct_text(text, alpha=1.0, beta=0.6, gamma=0.1, max_cands=40, enable_split_merge=True):
    tokens = tokenize(text)
    suggestions = []
    i = 0
    while i < len(tokens):
        w = tokens[i]
        if not is_word(w) or in_user_dict(w):
            i += 1; continue
        if not looks_suspicious(tokens, i):
            i += 1; continue

        cands = generate_candidates(w, topk=max_cands)
        scored = []
        prev_tok = tokens[i-1] if i-1 >= 0 else "<s>"
        next_tok = tokens[i+1] if i+1 < len(tokens) else "</s>"
        for c in cands:
            lp = mlm_logprob(tokens, i, c)
            b = bigram_score(prev_tok, c, next_tok)
            e = edit_cost(w.lower(), c.lower())
            freq_bonus = math.log1p(vocab_freq.get(c.lower(), 1))
            score = alpha*lp + b - beta*e + gamma*freq_bonus
            scored.append((c, score))

        if enable_split_merge:
            for kind, cand, lp in try_split_merge(tokens, i):
                e = edit_cost(w.lower(), cand.replace(" ", "").lower())
                freq_bonus = sum(math.log1p(vocab_freq.get(x,1)) for x in cand.split())
                score = alpha*lp - beta*e + gamma*freq_bonus
                scored.append((cand, score))

        if not scored:
            i += 1; continue

        best, best_score = max(scored, key=lambda x: x[1])
        if " " in best:
            left, right = best.split(" ", 1)
            left, right = preserve_case(w[:len(left)], left), preserve_case(w[len(left):], right)
            suggestions.append({"pos": i, "orig": w, "suggestion": best})
            tokens = tokens[:i] + [left, right] + tokens[i+1:]
            i += 2; continue

        if best.lower() != w.lower():
            final = preserve_case(w, best)
            suggestions.append({"pos": i, "orig": w, "suggestion": final})
            tokens[i] = final

        i += 1

    return detokenize(tokens), suggestions


# ===========================
# Step 16: Test samples
# ===========================
tests = [
    "I went too school withoout my bag today.",
    "He sayed that their going to thair home.",
    "Im very hapy to recieve the package.",
    "I wanto go home now.",
]
for t in tests:
    fixed, sugg = correct_text(t)
    print("\nINPUT :", t)
    print("OUTPUT:", fixed)
    print("SUGG  :", sugg[:5])


# ===========================
# Step 17: Corruption functions
# ===========================
NEIGHBORS = {
    'e': list('wsdr3'), 'r': list('edft4'), 't': list('rfgy5'), 'a': list('qwsz'),
    'i': list('ujko8'), 'o': list('i9pl'), 'n': list('bhjm'), 'l': list('kop;'),
}

def corrupt_word(w, p=0.15):
    if len(w) == 0: return w
    ops = []
    if random.random() < p: ops.append('delete')
    if random.random() < p: ops.append('insert')
    if random.random() < p: ops.append('sub')
    if random.random() < p and len(w) > 1: ops.append('transpose')
    w = list(w)
    for op in ops:
        if op == 'delete' and w:
            idx = random.randrange(len(w)); w.pop(idx)
        elif op == 'insert':
            idx = random.randrange(len(w)+1)
            base = w[min(idx, len(w)-1)].lower() if w else 'e'
            ch = random.choice(NEIGHBORS.get(base, list('aeiou')))
            w.insert(idx, ch)
        elif op == 'sub' and w:
            idx = random.randrange(len(w))
            ch = random.choice(NEIGHBORS.get(w[idx].lower(), list('aeiou')))
            w[idx] = ch
        elif op == 'transpose' and len(w) > 1:
            idx = random.randrange(len(w)-1)
            w[idx], w[idx+1] = w[idx+1], w[idx]
    return ''.join(w)

def corrupt_sentence(text, p_word=0.25):
    toks = tokenize(text)
    out = []
    for t in toks:
        if is_word(t) and random.random() < p_word:
            out.append(corrupt_word(t))
        else:
            out.append(t)
    return detokenize(out)


# ===========================
# Step 18: Evaluate accuracy
# ===========================
eval_lines = tests * 5
total = 0; exact = 0
for line in eval_lines:
    noisy = corrupt_sentence(line, p_word=0.3)
    fixed, _ = correct_text(noisy)
    total += 1
    exact += int(fixed.lower() == line.lower())
print(f"Exact sentence recovery: {exact}/{total} = {exact/total:.2%}")


# ===========================
# Step 19: Gradio UI
# ===========================
def ui_correct(text):
    fixed, sugg = correct_text(text)
    return fixed, json.dumps(sugg, indent=2, ensure_ascii=False)

with gr.Blocks() as demo:
    gr.Markdown("# ✍️ AI Autocorrect")
    inp = gr.Textbox(label="Input text", lines=3, value="I went too school withoout my bag today.")
    out = gr.Textbox(label="Corrected text", lines=3)
    sug = gr.Textbox(label="Suggestions (JSON)", lines=10)
    btn = gr.Button("Correct")
    btn.click(fn=ui_correct, inputs=inp, outputs=[out, sug])

demo.launch(share=False)
