"""
GPT from scratch — Decoder-only Transformer 언어모델
Tiny Shakespeare 데이터셋으로 character-level 텍스트 생성을 학습합니다.

Reference:
    - Vaswani et al., "Attention Is All You Need" (2017)
    - Radford et al., "Improving Language Understanding by Generative Pre-Training" (GPT-1, 2018)
    - Karpathy's nanoGPT (https://github.com/karpathy/nanoGPT)
"""

import math
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# matplotlib에 한글 폰트 적용 (Windows: Malgun Gothic, macOS: AppleGothic, Linux: NanumGothic)
_korean_fonts = ["Malgun Gothic", "NanumGothic", "NanumBarunGothic", "AppleGothic", "Noto Sans CJK KR", "Gulim"]
_available = {f.name for f in fm.fontManager.ttflist}
for _font in _korean_fonts:
    if _font in _available:
        mpl.rcParams["font.family"] = _font
        break
mpl.rcParams["axes.unicode_minus"] = False  # 마이너스 부호 깨짐 방지


# ───────────────────────────────────────────────────────────
# 1. 경로 / 시드 / 디바이스
# ───────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
torch.manual_seed(SEED)
if DEVICE == "cuda":
    torch.cuda.manual_seed_all(SEED)


# ───────────────────────────────────────────────────────────
# 2. 하이퍼파라미터
# ───────────────────────────────────────────────────────────
BLOCK_SIZE = 256        # context length (한 번에 보는 토큰 수)
BATCH_SIZE = 64
N_LAYER = 6             # Transformer block 개수
N_HEAD = 6              # multi-head 개수
N_EMBD = 384            # embedding dim (head_dim = 384/6 = 64)
DROPOUT = 0.2

MAX_ITERS = 5000
WARMUP_ITERS = 100
LR_MAX = 1e-3
LR_MIN = 1e-4
GRAD_CLIP = 1.0

EVAL_INTERVAL = 500
EVAL_ITERS = 100


# ───────────────────────────────────────────────────────────
# 3. 데이터 (Tiny Shakespeare, char-level)
# ───────────────────────────────────────────────────────────
def download_shakespeare():
    """Karpathy의 char-rnn 레포에서 Tiny Shakespeare 텍스트를 받아 로컬에 캐시."""
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    path = DATA_DIR / "tinyshakespeare.txt"
    if not path.exists():
        print(f"[data] downloading from {url} ...")
        urllib.request.urlretrieve(url, path)
    return path.read_text(encoding="utf-8")


text = download_shakespeare()
chars = sorted(set(text))
vocab_size = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for i, c in enumerate(chars)}


def encode(s):
    return [stoi[c] for c in s]


def decode(ids):
    return "".join(itos[int(i)] for i in ids)


data = torch.tensor(encode(text), dtype=torch.long)
n_train = int(0.9 * len(data))
train_data = data[:n_train]
val_data = data[n_train:]


def get_batch(split):
    """랜덤하게 시작 위치를 잡아 BATCH_SIZE개의 (x, y) 시퀀스를 만든다.
       x: t ~ t+BLOCK_SIZE-1
       y: t+1 ~ t+BLOCK_SIZE  (next-token target)
    """
    d = train_data if split == "train" else val_data
    ix = torch.randint(0, len(d) - BLOCK_SIZE - 1, (BATCH_SIZE,))
    x = torch.stack([d[i : i + BLOCK_SIZE] for i in ix])
    y = torch.stack([d[i + 1 : i + BLOCK_SIZE + 1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)


# ───────────────────────────────────────────────────────────
# 4. 모델 컴포넌트
# ───────────────────────────────────────────────────────────
class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention.

    - QKV는 한 번의 Linear로 묶어서 계산 후 split.
    - 미래 토큰을 보지 못하도록 lower-triangular 마스크 적용.
    - 마스크는 register_buffer로 저장 (state_dict에 들어가지만 학습되지는 않음).
    """

    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        assert n_embd % n_head == 0, "n_embd는 n_head로 나누어 떨어져야 합니다."
        self.n_head = n_head
        self.head_dim = n_embd // n_head

        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        # (1, 1, T, T) 형태의 lower-triangular 마스크
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size),
        )

    def forward(self, x, return_attn=False):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=-1)
        # (B, T, C) → (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # scaled dot-product
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, nh, T, T)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att_weights = att  # 시각화용 (dropout 적용 전)
        att = self.attn_drop(att)

        y = att @ v  # (B, nh, T, head_dim)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_drop(self.proj(y))

        if return_attn:
            return y, att_weights
        return y


class MLP(nn.Module):
    """ Position-wise feed-forward (4x expansion + GELU) """

    def __init__(self, n_embd, dropout):
        super().__init__()
        self.fc1 = nn.Linear(n_embd, 4 * n_embd)
        self.fc2 = nn.Linear(4 * n_embd, n_embd)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.fc2(F.gelu(self.fc1(x))))


class Block(nn.Module):
    """ Pre-LN Transformer block: x = x + Attn(LN(x));  x = x + MLP(LN(x)) """

    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    """
    Decoder-only Transformer 언어모델.

    구조:
        token_emb + pos_emb → dropout → [Block × N] → LayerNorm → Linear(vocab_size)
    """

    def __init__(self, vocab_size, block_size, n_layer, n_head, n_embd, dropout):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """주어진 컨텍스트 idx (B, T)에 이어서 max_new_tokens개의 토큰을 샘플링."""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-8)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_tok], dim=1)
        return idx

    def num_params(self):
        return sum(p.numel() for p in self.parameters())


# ───────────────────────────────────────────────────────────
# 5. 학습 유틸
# ───────────────────────────────────────────────────────────
@torch.no_grad()
def estimate_loss(model):
    """Train/Val 양쪽 split에서 EVAL_ITERS번 평균 loss를 측정."""
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def get_lr(it):
    """Linear warmup → cosine decay."""
    if it < WARMUP_ITERS:
        return LR_MAX * (it + 1) / WARMUP_ITERS
    if it > MAX_ITERS:
        return LR_MIN
    progress = (it - WARMUP_ITERS) / max(1, MAX_ITERS - WARMUP_ITERS)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return LR_MIN + coeff * (LR_MAX - LR_MIN)


@torch.no_grad()
def sample_text(model, prompt, max_new_tokens=300, temperature=1.0, top_k=None, greedy=False):
    """모델로 텍스트를 한 번 생성해서 string으로 반환."""
    model.eval()
    idx = torch.tensor(encode(prompt), dtype=torch.long, device=DEVICE).unsqueeze(0)
    if greedy:
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -model.block_size :]
            logits, _ = model(idx_cond)
            next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            idx = torch.cat([idx, next_tok], dim=1)
    else:
        idx = model.generate(idx, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)
    model.train()
    return decode(idx[0].tolist())


# ───────────────────────────────────────────────────────────
# 6. 시각화
# ───────────────────────────────────────────────────────────
def plot_dataset_overview():
    """01: 데이터셋 개요 — 문자 빈도 + 통계 + 샘플 텍스트."""
    fig = plt.figure(figsize=(15, 8))

    # (a) 상위 30개 문자 빈도
    freq = Counter(text).most_common(30)
    chs, cnts = zip(*freq)
    chs_disp = ["\\n" if c == "\n" else "_" if c == " " else c for c in chs]

    ax1 = plt.subplot2grid((2, 2), (0, 0))
    ax1.bar(range(len(chs)), cnts, color="steelblue")
    ax1.set_xticks(range(len(chs)))
    ax1.set_xticklabels(chs_disp, fontsize=11, fontweight="bold")
    ax1.tick_params(axis="y", labelsize=11)
    ax1.set_title("Top-30 Character Frequency", fontsize=14, fontweight="bold")
    ax1.set_ylabel("Count", fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.3, axis="y")

    # (b) 데이터셋 통계
    ax2 = plt.subplot2grid((2, 2), (1, 0))
    ax2.axis("off")
    stats_lines = [
        f"Vocab size      : {vocab_size}",
        f"Total chars     : {len(text):,}",
        f"Train chars     : {len(train_data):,}",
        f"Val chars       : {len(val_data):,}",
        f"Block (context) : {BLOCK_SIZE}",
        f"Tokenizer       : char-level (no BPE)",
    ]
    ax2.text(0.0, 0.85, "\n".join(stats_lines),
             fontsize=13, family="monospace", fontweight="bold", va="top")
    ax2.set_title("Dataset Statistics", loc="left", fontsize=14, fontweight="bold")

    # (c) 샘플 텍스트 800자
    ax3 = plt.subplot2grid((2, 2), (0, 1), rowspan=2)
    ax3.axis("off")
    ax3.set_title("Sample Text (first 800 chars)", loc="left", fontsize=14, fontweight="bold")
    ax3.text(0.0, 1.0, text[:800], fontsize=10, family="monospace", va="top")

    plt.suptitle("Tiny Shakespeare Dataset Overview", fontsize=16, fontweight="bold", y=1.00)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "01_dataset_overview.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("[viz] saved 01_dataset_overview.png")


def _get_attention_at(model, sample, block_idx):
    """주어진 입력에 대해 특정 block의 attention weights를 반환. shape: (n_head, T, T)."""
    T = len(sample)
    seq = torch.tensor(encode(sample), dtype=torch.long, device=DEVICE).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        x = model.tok_emb(seq) + model.pos_emb(torch.arange(T, device=DEVICE))
        x = model.drop(x)
        for i in range(block_idx):
            x = model.blocks[i](x)
        x_ln = model.blocks[block_idx].ln1(x)
        _, att = model.blocks[block_idx].attn(x_ln, return_attn=True)
    return att[0].cpu().numpy()  # (n_head, T, T)


def plot_attention(model):
    """02: 학습된 attention 패턴을 6개 head × 2 layer (early vs late)로 비교."""
    sample = "ROMEO: Speak softly,"  # 20자
    T = len(sample)

    # Block 0 (early)와 Block 5 (late) 각각의 6개 head
    att_early = _get_attention_at(model, sample, block_idx=0)   # (6, T, T)
    att_late = _get_attention_at(model, sample, block_idx=N_LAYER - 1)  # (6, T, T)

    fig, axes = plt.subplots(2, 6, figsize=(22, 8.5))

    for h in range(N_HEAD):
        # 위쪽 행: Block 0
        ax = axes[0, h]
        im = ax.imshow(att_early[h], cmap="viridis", aspect="equal", vmin=0, vmax=1)
        ax.set_title(f"Head {h}", fontsize=13, fontweight="bold")
        ax.set_xticks(range(T))
        ax.set_xticklabels(list(sample), fontsize=9, fontweight="bold")
        ax.set_yticks(range(T))
        ax.set_yticklabels(list(sample), fontsize=9, fontweight="bold")
        if h == 0:
            ax.set_ylabel("Block 0\n(Early)", fontsize=14, fontweight="bold")

        # 아래쪽 행: Block 5
        ax = axes[1, h]
        ax.imshow(att_late[h], cmap="viridis", aspect="equal", vmin=0, vmax=1)
        ax.set_xticks(range(T))
        ax.set_xticklabels(list(sample), fontsize=9, fontweight="bold")
        ax.set_yticks(range(T))
        ax.set_yticklabels(list(sample), fontsize=9, fontweight="bold")
        if h == 0:
            ax.set_ylabel(f"Block {N_LAYER - 1}\n(Late)", fontsize=14, fontweight="bold")

    # 공통 컬러바
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.015, pad=0.02)
    cbar.set_label("attention weight", fontsize=12, fontweight="bold")
    cbar.ax.tick_params(labelsize=11)

    plt.suptitle(
        "Multi-Head Attention Patterns  —  Early Layer (top) vs Late Layer (bottom)\n"
        f'input: "{sample}"',
        fontsize=15, fontweight="bold", y=0.99,
    )
    plt.savefig(RESULTS_DIR / "02_attention_visualization.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("[viz] saved 02_attention_visualization.png")


def plot_training_curve(history):
    """03: Train/Val Loss + LR 스케줄 + 어노테이션 (val 최저점, overfitting zone, 무작위 baseline)."""
    iters = np.array([h["iter"] for h in history])
    train_losses = np.array([h["train"] for h in history])
    val_losses = np.array([h["val"] for h in history])

    # LR 스케줄을 부드럽게 표시하기 위해 모든 iter에서 계산
    lr_iters_dense = np.arange(0, MAX_ITERS + 1)
    lrs_dense = np.array([get_lr(int(it)) for it in lr_iters_dense])

    min_val_idx = int(np.argmin(val_losses))
    min_val_iter = int(iters[min_val_idx])
    min_val_loss = float(val_losses[min_val_idx])

    fig, ax = plt.subplots(figsize=(12, 7))

    # 무작위 추측 baseline (ln(vocab))
    random_baseline = math.log(vocab_size)
    ax.axhline(y=random_baseline, color="gray", linestyle=":", alpha=0.7, linewidth=1.5)
    ax.text(
        iters.max() * 0.50, random_baseline + 0.10,
        f"무작위 추측 baseline = ln({vocab_size}) " r"$\approx$" f" {random_baseline:.2f}",
        color="gray", fontsize=12, fontweight="bold", style="italic",
    )

    # Overfitting zone (val 최저점 이후)
    ax.axvspan(min_val_iter, iters.max(), alpha=0.10, color="red", label="_nolegend_")
    ax.text(
        min_val_iter + (iters.max() - min_val_iter) / 2,
        train_losses.min() + 0.30,
        "Overfitting Zone\n(train ↓, val ↑)",
        ha="center", fontsize=13, fontweight="bold", color="darkred", alpha=0.9, style="italic",
    )

    # Loss 곡선 (왼쪽 축)
    line_train, = ax.plot(iters, train_losses, label="Train Loss", color="steelblue",
                          marker="o", linewidth=2.5, markersize=8)
    line_val, = ax.plot(iters, val_losses, label="Val Loss", color="crimson",
                        marker="s", linewidth=2.5, markersize=8)

    # Val 최저점 강조
    ax.scatter([min_val_iter], [min_val_loss], color="green", s=260, zorder=5,
               edgecolor="darkgreen", linewidth=2.5, label="_nolegend_")
    ax.annotate(
        f"Val Loss 최저점\n{min_val_loss:.3f} @ iter {min_val_iter}\n→ 이상적 early stopping",
        xy=(min_val_iter, min_val_loss),
        xytext=(min_val_iter + 1100, min_val_loss + 0.7),
        fontsize=12, color="darkgreen", fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="darkgreen", lw=2),
        bbox=dict(boxstyle="round,pad=0.5", fc="lightyellow", ec="darkgreen", lw=1.5),
    )

    ax.set_xlabel("Iteration", fontsize=13, fontweight="bold")
    ax.set_ylabel("Cross-Entropy Loss", fontsize=13, fontweight="bold")
    ax.tick_params(axis="both", labelsize=11)
    ax.set_title("Training Curve  —  Loss & LR Schedule",
                 fontsize=15, fontweight="bold")
    ax.grid(True, alpha=0.3)

    # 보조 축: Learning Rate 스케줄
    ax2 = ax.twinx()
    line_lr, = ax2.plot(lr_iters_dense, lrs_dense, label="Learning Rate",
                        color="darkgoldenrod", linestyle="--", linewidth=2, alpha=0.75)
    ax2.set_ylabel("Learning Rate", fontsize=13, fontweight="bold", color="darkgoldenrod")
    ax2.tick_params(axis="y", labelsize=11, colors="darkgoldenrod")
    ax2.set_ylim(0, lrs_dense.max() * 1.15)
    # LR 축의 spine 색상 맞춤
    ax2.spines["right"].set_color("darkgoldenrod")
    # LR 축 위 LR 최대치 어노테이션 (워밍업 끝나는 지점)
    ax2.annotate(
        f"warmup 종료\nLR={LR_MAX:.0e}",
        xy=(WARMUP_ITERS, LR_MAX),
        xytext=(WARMUP_ITERS + 300, LR_MAX * 1.05),
        fontsize=10, color="darkgoldenrod", fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="darkgoldenrod", lw=1.2),
    )

    # 두 축의 line을 합쳐서 하나의 legend로
    lines = [line_train, line_val, line_lr]
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc="upper right", fontsize=12)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "03_training_curve.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("[viz] saved 03_training_curve.png")


def _text_height_inches(txt, font_pt=11.0, line_spacing=1.3):
    """Monospace 텍스트의 높이를 inch 단위로 추정 (line_count × pt × spacing / 72)."""
    n_lines = txt.count("\n") + 1
    return n_lines * font_pt * line_spacing / 72.0


def plot_generation_progression(snapshots):
    """04: 학습 진행에 따른 생성 텍스트 변화. 각 패널 높이를 텍스트 라인 수에 맞춰 동적 계산."""
    # 각 iter에 대한 짧은 의미 라벨 + 컬러
    labels = {
        0:    ("학습 전 (랜덤 초기화)",                "#cc2222"),
        2500: ("형식 학습 완료 (이름: 대사 패턴)",       "#cc8800"),
        5000: ("긴 문장 + 운율 + 셰익스피어 스타일",     "#1f7f1f"),
    }
    last_iter = max(s[0] for s in snapshots)
    if last_iter not in labels:
        labels[last_iter] = ("긴 문장 + 운율 + 셰익스피어 스타일", "#1f7f1f")

    # 패널 높이 = 본문 텍스트 높이 + 헤더 영역
    panel_heights = [_text_height_inches(txt) + 0.6 for _, txt in snapshots]
    total_height = sum(panel_heights) + 0.7  # for suptitle and margins

    fig, axes = plt.subplots(
        len(snapshots), 1,
        figsize=(14, total_height),
        gridspec_kw={"height_ratios": panel_heights, "hspace": 0.45},
    )
    if len(snapshots) == 1:
        axes = [axes]

    for ax, (it, txt) in zip(axes, snapshots):
        ax.axis("off")
        desc, color = labels.get(it, ("", "#000000"))
        # 헤더: "Iteration NNNN  —  설명" 한 줄로
        header = f"Iteration {it}  —  {desc}"
        ax.set_title(header, loc="left", fontsize=15, fontweight="bold", color=color, pad=6)
        # 본문 텍스트
        ax.text(0.0, 1.0, txt, fontsize=11, family="monospace", va="top",
                transform=ax.transAxes)

    plt.suptitle("Generated Samples Over Training", fontsize=16, fontweight="bold", y=0.998)
    plt.savefig(RESULTS_DIR / "04_generation_progression.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("[viz] saved 04_generation_progression.png")


def plot_sampling_comparison(model, prompt="ROMEO:"):
    """05: 동일 프롬프트, 다른 샘플링 전략 비교. 각 전략에 특성 태그를 부착."""
    # (label, temp, top_k, greedy, characterization, color)
    cfgs = [
        ("Greedy (argmax)",  None, None, True,
         "결정론적, 안정적이지만 반복 발생",       "#7a3b3b"),
        ("Temperature 0.7",   0.7, None, False,
         "보수적, 짧고 안정된 대사 위주",          "#3b6b8a"),
        ("Temperature 1.0",   1.0, None, False,
         "균형, 가장 자연스러운 셰익스피어 톤",    "#1f7f1f"),
        ("Temperature 1.4",   1.4, None, False,
         "발산, 새 단어 조합 등장하지만 문법 깨짐", "#b8860b"),
        ("Top-k = 10",        1.0, 10,   False,
         "안정성과 다양성의 균형",                  "#5b3b8a"),
    ]

    samples = []
    for label, temp, topk, greedy, desc, color in cfgs:
        out = sample_text(
            model, prompt=prompt,
            max_new_tokens=200,
            temperature=temp if temp is not None else 1.0,
            top_k=topk, greedy=greedy,
        )
        samples.append((label, desc, color, out))

    panel_heights = [_text_height_inches(s[3]) + 0.6 for s in samples]
    total_height = sum(panel_heights) + 0.7

    fig, axes = plt.subplots(
        len(samples), 1,
        figsize=(14, total_height),
        gridspec_kw={"height_ratios": panel_heights, "hspace": 0.5},
    )
    if len(samples) == 1:
        axes = [axes]

    for ax, (label, desc, color, txt) in zip(axes, samples):
        ax.axis("off")
        # 헤더: "라벨  —  설명" 한 줄로
        header = f"{label}   —   {desc}"
        ax.set_title(header, loc="left", fontsize=15, fontweight="bold", color=color, pad=6)
        # 생성 텍스트
        ax.text(0.0, 1.0, txt, fontsize=11, family="monospace", va="top",
                transform=ax.transAxes)

    plt.suptitle(
        f'Sampling Strategy Comparison  (prompt: "{prompt}")',
        fontsize=16, fontweight="bold", y=0.998,
    )
    plt.savefig(RESULTS_DIR / "05_sampling_comparison.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("[viz] saved 05_sampling_comparison.png")


# ───────────────────────────────────────────────────────────
# 7. 메인 파이프라인
# ───────────────────────────────────────────────────────────
CHECKPOINT_PATH = DATA_DIR / "checkpoint.pt"


def build_model():
    return GPT(
        vocab_size=vocab_size,
        block_size=BLOCK_SIZE,
        n_layer=N_LAYER,
        n_head=N_HEAD,
        n_embd=N_EMBD,
        dropout=DROPOUT,
    ).to(DEVICE)


def train_loop(model):
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR_MAX, betas=(0.9, 0.99))
    history = []
    snapshots = []

    # iter 0: 학습 전 (랜덤 초기화) 샘플
    snapshots.append((0, sample_text(model, prompt="\n", max_new_tokens=250, temperature=1.0)))

    print("[train] start")
    t0 = time.time()
    for it in range(MAX_ITERS + 1):
        lr_now = get_lr(it)
        for g in optimizer.param_groups:
            g["lr"] = lr_now

        if it % EVAL_INTERVAL == 0:
            losses = estimate_loss(model)
            elapsed = time.time() - t0
            print(f"[iter {it:5d}] train={losses['train']:.4f}  val={losses['val']:.4f}  "
                  f"lr={lr_now:.2e}  ({elapsed:.1f}s)", flush=True)
            history.append({"iter": it, "train": losses["train"], "val": losses["val"]})

        if it == MAX_ITERS:
            break

        X, Y = get_batch("train")
        _, loss = model(X, Y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        if it == MAX_ITERS // 2:
            snapshots.append((it, sample_text(model, prompt="\n", max_new_tokens=250, temperature=1.0)))

    snapshots.append((MAX_ITERS, sample_text(model, prompt="\n", max_new_tokens=250, temperature=1.0)))

    print(f"[train] total time: {(time.time() - t0) / 60:.1f} min")
    return history, snapshots


def main():
    """
    실행 옵션:
        python src/main.py             # checkpoint 있으면 load, 없으면 학습 후 저장
        python src/main.py --retrain   # 강제로 다시 학습 (기존 checkpoint 무시)
        python src/main.py --viz-only  # checkpoint만 load, 학습 스킵 (없으면 에러)
    """
    force_retrain = "--retrain" in sys.argv
    viz_only = "--viz-only" in sys.argv

    print(f"[info] device          : {DEVICE}")
    print(f"[info] vocab size      : {vocab_size}")
    print(f"[info] train tokens    : {len(train_data):,}")
    print(f"[info] val   tokens    : {len(val_data):,}", flush=True)

    # 시각화 01: 데이터셋 개요 (학습 무관)
    plot_dataset_overview()

    model = build_model()
    print(f"[info] model params    : {model.num_params() / 1e6:.2f} M", flush=True)

    if CHECKPOINT_PATH.exists() and not force_retrain:
        print(f"[load] checkpoint found at {CHECKPOINT_PATH.name}, skipping training", flush=True)
        ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model"])
        history = ckpt["history"]
        snapshots = ckpt["snapshots"]
    else:
        if viz_only:
            raise FileNotFoundError(f"--viz-only 모드인데 checkpoint가 없습니다: {CHECKPOINT_PATH}")
        history, snapshots = train_loop(model)
        torch.save(
            {"model": model.state_dict(), "history": history, "snapshots": snapshots},
            CHECKPOINT_PATH,
        )
        print(f"[save] checkpoint saved to {CHECKPOINT_PATH.name}", flush=True)

    # 시각화 02~05
    plot_attention(model)
    plot_training_curve(history)
    plot_generation_progression(snapshots)
    plot_sampling_comparison(model, prompt="ROMEO:")

    print("[done] all visualizations saved to results/")


if __name__ == "__main__":
    main()
