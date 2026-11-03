"""Build the project page (docs/index.html) and its figures from the committed results.

Every number on the page is read from analysis/numbers.tex, analysis/paired.json or the raw
verdicts in data/, the same files the paper is built from. Nothing calls a model. Writes
docs/index.html, docs/figures/*.svg and docs/social-card.png.

Run from the repository root:  python docs/make_page.py
"""

from __future__ import annotations

import html
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
sys.path.insert(0, str(ROOT / "analysis"))

import make_figures as mf  # noqa: E402  (sets the paper's rcParams; overridden below)
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

FONT_DIR = Path("/usr/local/texlive/2025/texmf-dist/fonts/truetype/ascender/opensans")

UU_RED = "#C00A35"
INK = "#1a1a1a"
GRAY = "#b9bcc2"
BAND = "#fff0ba"

OUTCOME = "outcome:qwen2.5:14b"
STEP = "step:qwen2.5:14b"

PLAIN_TYPE = {
    "wrong_tool": "used the wrong tool",
    "hallucinated_argument": "made up a tool input",
    "skipped_precondition": "skipped the eligibility check",
    "ignored_observation": "ignored a tool result",
    "unsupported_claim": "invented a promise in the reply",
    "premature_stop": "stopped too early",
}
PLAIN_JUDGE = {
    "programmatic": "Rule checker",
    "outcome:qwen2.5:14b": "LLM, reads the reply only",
    "step:qwen2.5:14b": "LLM, checks each step",
    "step:llama3.1:8b": "Small LLM (8B), checks each step",
    "selfcons3:qwen2.5:14b": "LLM, checks each step, 3 votes",
}

URL = "https://mohammadi.cv/trajectory-judge/"
REPO = "https://github.com/mohammadi-hadi/trajectory-judge"
ARXIV = "https://arxiv.org/abs/2609.00038"
CV = "https://mohammadi.cv/assets/docs/HadiMohammadiCV.pdf"
NO_DATE = {"Date": None}  # keeps the SVGs byte-identical across rebuilds


def setup_fonts() -> None:
    family = "DejaVu Sans"
    if FONT_DIR.is_dir():
        for f in FONT_DIR.glob("OpenSans-*.ttf"):
            font_manager.fontManager.addfont(str(f))
        family = "Open Sans"
    plt.rcParams.update(
        {
            "font.family": family,
            "font.size": 12,
            "axes.labelsize": 11.5,
            "xtick.labelsize": 11.5,
            "ytick.labelsize": 12,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "path",
            "svg.hashsalt": "trajectory-judge",
            "figure.facecolor": "white",
        }
    )


def macros() -> dict[str, str]:
    """The paper's macro file as plain strings: real minus signs, unescaped percent signs."""
    out = {}
    for line in (ROOT / "analysis" / "numbers.tex").read_text().splitlines():
        m = re.match(r"\\newcommand\{\\(\w+)\}\{(.*)\}\s*$", line)
        if m:
            out[m.group(1)] = m.group(2).replace("\\ensuremath{-}", "\u2212").replace("\\%", "%")
    return out


def fig_by_type(trajectories, verdicts, narrow: bool = False) -> None:
    """Per mistake type: how often each 14B judge flags the correct twin (ring) and the mistake (dot).

    narrow=True stacks the two panels for phones, where side by side they shrink to unreadable.
    """
    rates = mf.paired_rates_by_type(trajectories, verdicts, [OUTCOME, STEP])
    ys = [5.5, 4.5, 3.5, 2.5, 1.0, 0.0]
    if narrow:
        fig, axes = plt.subplots(2, 1, figsize=(4.0, 7.4), gridspec_kw={"hspace": 0.55})
    else:
        fig, axes = plt.subplots(
            1, 2, figsize=(8.4, 4.2), sharey=True, gridspec_kw={"wspace": 0.42}
        )
    titles = {OUTCOME: "Reads the reply only", STEP: "Checks each step"}
    for ax, j in zip(axes, [OUTCOME, STEP], strict=True):
        color = UU_RED if j == OUTCOME else INK
        ax.axhspan(1.9, 6.15, color=BAND, zorder=0, lw=0)
        for y, (ftype, _) in zip(ys, mf.BY_TYPE_ROWS, strict=True):
            fault, parent, _ = rates[j][ftype]
            ax.plot([parent, fault], [y, y], color=color, lw=3, zorder=2, solid_capstyle="round")
            ax.scatter(
                [parent], [y], s=120, facecolors="white", edgecolors=INK, linewidths=1.6, zorder=3
            )
            ax.scatter([fault], [y], s=60, color=color, zorder=4)
            d = fault - parent
            ax.text(
                1.13,
                y,
                mf.signed(d),
                va="center",
                ha="left",
                fontsize=12,
                fontweight="bold",
                color=color if abs(d) > 0.05 else "#777777",
            )
        ax.set_xlim(-0.06, 1.06)
        ax.set_ylim(-0.6, 6.75)
        ax.set_xticks([0, 0.5, 1])
        ax.set_xticklabels(["0", "0.5", "1"])
        ax.set_xlabel("share of runs flagged")
        ax.set_title(titles[j], fontsize=12.5, fontweight="bold", color=color, pad=8)
        ax.text(1.13, 6.35, "Δ", ha="left", va="center", fontsize=12, fontweight="bold")
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    for ax in axes if narrow else axes[:1]:
        ax.set_yticks(ys)
        ax.set_yticklabels([PLAIN_TYPE[t] for t, _ in mf.BY_TYPE_ROWS])
        ax.text(-0.04, 6.38, "reply unchanged", fontsize=11, style="italic", va="center")
        ax.text(-0.04, 1.62, "reply changed", fontsize=11, style="italic", va="center")
    name = "by_type_narrow.svg" if narrow else "by_type.svg"
    fig.savefig(DOCS / "figures" / name, bbox_inches="tight", pad_inches=0.08, metadata=NO_DATE)
    plt.close(fig)


def fig_judges(ci, paired, narrow: bool = False) -> None:
    """Recall against Δ for the five judges, with false alarms on the 100 correct runs.

    narrow=True puts each judge's name above its bars instead of beside them, for phones.
    """
    pub = paired["published"]
    order = ["programmatic", OUTCOME, STEP, "selfcons3:qwen2.5:14b", "step:llama3.1:8b"]
    fig, ax = plt.subplots(figsize=(4.2, 5.6) if narrow else (7.6, 3.9))
    h = 0.38
    step = 1.45 if narrow else 1.0
    for i, j in enumerate(order):
        y = (len(order) - 1 - i) * step
        if narrow:
            ax.text(0, y + h + 0.1, PLAIN_JUDGE[j], va="bottom", fontsize=12, color=INK)
        rec = ci["judges"][j]["recall"]["point"]
        dlt = pub["paired"][j]["all"]["delta"]
        color = UU_RED if j == OUTCOME else INK
        ax.barh(y + h / 2, rec, height=h, color=GRAY, zorder=2)
        ax.barh(y - h / 2, max(dlt, 0), height=h, color=color, zorder=2)
        ax.text(
            rec + 0.02, y + h / 2 + 0.02, f"{rec:.2f}", va="center", fontsize=11, color="#555555"
        )
        ax.text(
            max(dlt, 0) + 0.02,
            y - h / 2 - 0.02,
            mf.signed(dlt),
            va="center",
            fontsize=11,
            fontweight="bold",
            color=color,
        )
        fa = pub["clean_flags"][j]
        ax.text(
            1.55,
            y,
            f"{fa}/100",
            va="center",
            ha="right",
            fontsize=12,
            fontweight="bold",
            color=UU_RED if fa >= 10 else INK,
        )
    top = (len(order) - 1) * step + (0.75 if narrow else 0.65)
    ax.text(1.55, top, "false alarms", ha="right", va="bottom", fontsize=11)
    if narrow:
        ax.set_yticks([])
        ax.set_ylim(-0.6, top + 0.35)
    else:
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([PLAIN_JUDGE[j] for j in reversed(order)])
    ax.set_xlim(0, 1.56)
    ax.set_xticks([0, 0.5, 1])
    ax.set_xticklabels(["0", "0.5", "1"])
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    handles = [
        Patch(color=GRAY, label="recall (share flagged)"),
        Patch(color=INK, label="Δ (real catches)"),
    ]
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.42, -0.08 if narrow else -0.12),
        ncol=1 if narrow else 2,
        frameon=False,
        fontsize=11,
        columnspacing=1.2,
        handlelength=1.4,
    )
    name = "judges_narrow.svg" if narrow else "judges.svg"
    fig.savefig(DOCS / "figures" / name, bbox_inches="tight", pad_inches=0.08, metadata=NO_DATE)
    plt.close(fig)


def social_card(m: dict[str, str]) -> None:
    """The 1200 x 630 preview image that link previews show."""
    fig = plt.figure(figsize=(12, 6.3), dpi=100)
    fig.patch.set_facecolor("black")
    fig.text(0.06, 0.80, "trajectory-judge", color="white", fontsize=44, fontweight="bold")
    fig.text(
        0.06,
        0.66,
        "What outcome-only LLM judges miss on agent trajectories",
        color="white",
        fontsize=24,
    )
    fig.text(
        0.06,
        0.47,
        "If a mistake does not change the agent's final reply,\n"
        "a judge that reads only the reply cannot catch it.",
        color="#FFCD00",
        fontsize=22,
        linespacing=1.4,
    )
    fig.text(
        0.06,
        0.20,
        f"Reply-only judge: Δ = {m['tjAblAPairReplySame']} on hidden mistakes",
        color="white",
        fontsize=19,
    )
    fig.text(
        0.06,
        0.12,
        "Same model, checking each step: every hidden mistake caught",
        color="white",
        fontsize=19,
    )
    fig.text(
        0.94,
        0.05,
        "NeurIPS 2026 Workshop · arXiv 2609.00038",
        color="#bbbbbb",
        fontsize=15,
        ha="right",
    )
    fig.savefig(DOCS / "social-card.png", dpi=100, facecolor="black")
    plt.close(fig)


def hidden_counts(trajectories, verdicts, paired) -> dict[str, int]:
    """Counts the page states that numbers.tex does not carry directly."""
    hidden_types = {f for f, _ in mf.BY_TYPE_ROWS[: mf.N_REPLY_UNCHANGED]}
    same = [
        t
        for t in trajectories
        if t["label"]["faulty"] and t["label"]["failure_type"] in hidden_types
    ]
    return {
        "hidden_n": len(same),
        "hidden_caught": sum(bool(verdicts[STEP][t["trajectory_id"]]["faulty"]) for t in same),
        "step_clean_flags": paired["published"]["clean_flags"][STEP],
        "outcome_clean_flags": paired["published"]["clean_flags"][OUTCOME],
    }


CSS = """
  :root {
    color-scheme: light;
    --page: #f9f9f7; --surface: #fcfcfb;
    --ink: #0b0b0b; --ink-2: #52514e; --muted: #898781;
    --grid: #e1e0d9; --border: rgba(11,11,11,0.10);
    --accent: #2a78d6; --reply: #c00a35; --seen: #ffe680; --seen-ink: #0b0b0b;
    --tile: #fff4cc;
  }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
      color-scheme: dark;
      --page: #0d0d0d; --surface: #1a1a19;
      --ink: #ffffff; --ink-2: #c3c2b7; --muted: #898781;
      --grid: #2c2c2a; --border: rgba(255,255,255,0.10);
      --accent: #3987e5; --reply: #ff6b86; --seen: #e6c200; --seen-ink: #0b0b0b;
      --tile: #2a2614;
    }
  }
  :root[data-theme="dark"] {
    color-scheme: dark;
    --page: #0d0d0d; --surface: #1a1a19;
    --ink: #ffffff; --ink-2: #c3c2b7; --muted: #898781;
    --grid: #2c2c2a; --border: rgba(255,255,255,0.10);
    --accent: #3987e5; --reply: #ff6b86; --seen: #e6c200; --seen-ink: #0b0b0b;
    --tile: #2a2614;
  }
  * { box-sizing: border-box; margin: 0; }
  body {
    background: var(--page); color: var(--ink);
    font: 15px/1.6 system-ui, -apple-system, "Segoe UI", sans-serif;
    padding: 40px 16px 60px;
  }
  main { max-width: 880px; margin: 0 auto; }
  h1 { font-size: 26px; letter-spacing: -0.02em; }
  h1 a { color: inherit; text-decoration: none; }
  .tagline { color: var(--ink-2); font-size: 17px; margin: 6px 0 8px; }
  .status { color: var(--muted); font-size: 13px; margin-bottom: 14px; }
  .links { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 30px; }
  .links a {
    border: 1px solid var(--border); background: var(--surface); border-radius: 999px;
    padding: 5px 13px; font-size: 13.5px; text-decoration: none; color: var(--ink);
  }
  .links a:hover { border-color: var(--accent); }
  h2 { font-size: 18px; margin: 40px 0 12px; }
  p { margin: 0 0 12px; }
  .panel {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 10px; padding: 16px 18px; margin: 14px 0;
  }
  .figure { background: #ffffff; border: 1px solid var(--border); border-radius: 10px;
            padding: 12px; margin: 14px 0; }
  .figure img { display: block; width: 100%; height: auto; }
  .caption { color: var(--ink-2); font-size: 14px; }
  .reply { color: var(--reply); font-weight: 600; }
  .step { font-weight: 600; }
  .scroll { overflow-x: auto; }
  .run { display: flex; align-items: center; gap: 6px; min-width: 640px; margin: 4px 0; }
  @media (max-width: 600px) { .run { flex-wrap: wrap; min-width: 0; row-gap: 6px; } }
  .runlabel { font-size: 13px; font-weight: 600; margin: 10px 0 2px; }
  .chip {
    border: 1px solid var(--grid); border-radius: 6px; padding: 4px 9px; font-size: 13px;
    background: var(--surface); white-space: nowrap;
  }
  .chip.seen { background: var(--seen); color: var(--seen-ink); border-color: transparent; }
  .chip.gone { text-decoration: line-through; color: var(--muted); border-style: dashed; }
  .chip.bad { border: 2px solid var(--reply); }
  .arrow { color: var(--muted); font-size: 12px; }
  table { border-collapse: collapse; width: 100%; font-size: 14px; }
  th, td { padding: 8px 12px; border-bottom: 1px solid var(--grid); text-align: center;
           font-variant-numeric: tabular-nums; }
  th:first-child, td:first-child { text-align: left; }
  th { color: var(--muted); font-weight: 600; }
  tr:last-child td { border-bottom: none; }
  .tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px;
           margin: 14px 0; }
  .tile { background: var(--tile); border-radius: 10px; padding: 14px; text-align: center; }
  .tile b { display: block; font-size: 30px; line-height: 1.15; }
  .tile span { font-size: 13px; color: var(--ink-2); }
  .grid2 td.num { font-size: 22px; font-weight: 700; }
  .grid2 td small { display: block; font-size: 12px; font-weight: 400; color: var(--muted); }
  .quote { border-left: 3px solid var(--reply); padding: 4px 0 4px 14px; margin: 10px 0; }
  ol { padding-left: 22px; }
  ol li { margin: 6px 0; }
  pre { background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
        padding: 12px 14px; overflow-x: auto; font-size: 13px; }
  code { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
  a { color: var(--accent); }
  .about { display: flex; flex-wrap: wrap; gap: 6px 18px; align-items: baseline; }
  footer { margin-top: 44px; color: var(--muted); font-size: 13px; }
"""


def page(m: dict[str, str], c: dict[str, int], has_poster: bool) -> str:
    amount = m["tjPairExAmount"]
    reply = f"reply: \u201cI have refunded EUR {amount}\u2026\u201d"
    steps = ["customer", "order", "policy", "eligibility", "refund"]

    def run(broken: bool) -> str:
        chips = ['<span class="chip seen">request</span>']
        for s in steps:
            cls = "chip"
            if broken and s == "eligibility":
                cls += " gone"
            if broken and s == "refund":
                cls += " bad"
            chips.append(f'<span class="{cls}">{s}</span>')
        chips.append(f'<span class="chip seen">{html.escape(reply)}</span>')
        return '<div class="run">' + '<span class="arrow">\u2192</span>'.join(chips) + "</div>"

    poster_link = '<a href="poster.pdf">Poster (PDF)</a>' if has_poster else ""
    bibtex = (
        "@inproceedings{mohammadi2026trajectoryjudge,\n"
        "  author        = {Mohammadi, Hadi},\n"
        "  title         = {trajectory-judge: What Outcome-Only LLM Judges Miss on Agent Trajectories},\n"
        "  booktitle     = {NeurIPS 2026 Workshop: Who Verifies the Agents? Toward Reliable Agent Development},\n"
        "  year          = {2026},\n"
        "  eprint        = {2609.00038},\n"
        "  archivePrefix = {arXiv},\n"
        "  url           = {https://arxiv.org/abs/2609.00038}\n"
        "}"
    )
    title = "trajectory-judge: what outcome-only LLM judges miss on agent trajectories"
    desc = (
        "If a mistake does not change an AI agent's final reply, a judge that reads only the "
        "reply cannot catch it. A test bed with exact labels, and what recall hides."
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
  <link rel="canonical" href="{URL}">
  <meta property="og:site_name" content="Hadi Mohammadi">
  <meta property="og:image" content="{URL}social-card.png">
  <meta name="twitter:card" content="summary_large_image">
  <meta name="twitter:title" content="{html.escape(title)}">
  <meta name="twitter:description" content="{html.escape(desc)}">
  <meta name="twitter:image" content="{URL}social-card.png">
  <meta name="author" content="Hadi Mohammadi">
<meta name="description" content="{html.escape(desc)}">
<meta property="og:title" content="{html.escape(title)}">
<meta property="og:description" content="{html.escape(desc)}">
<meta property="og:url" content="{URL}">
<meta property="og:type" content="website">
<!-- generated by docs/make_page.py from analysis/numbers.tex, analysis/paired.json and data/ -->
<style>{CSS}</style>
</head>
<body>
<main>
  <h1><a href="{REPO}">trajectory-judge</a></h1>
  <p class="tagline">If a mistake does not change an AI agent's final reply, a judge that reads
  only the reply cannot catch it.</p>
  <p class="status">Hadi Mohammadi, Utrecht University &middot; NeurIPS 2026 Workshop: Who
  Verifies the Agents? Poster, Sydney, Saturday 12 December</p>
  <nav class="links">
    <a href="{ARXIV}">Paper (arXiv)</a>
    <a href="{REPO}">Code &amp; data</a>
    {poster_link}
    <a href="https://pypi.org/project/trajectory-judge/">pip install trajectory-judge</a>
    <a href="https://doi.org/10.5281/zenodo.21797926">Zenodo</a>
    <a href="{CV}">My CV</a>
  </nav>

  <h2>Same reply. One run has a mistake. Can the judge tell?</h2>
  <p>An AI agent uses tools to help a customer. Another AI, the <b>judge</b>, grades its work.
  To test a judge, people plant mistakes and count how many it flags. That count is called
  <b>recall</b>.</p>
  <div class="panel">
    <div class="scroll">
      <div class="runlabel">Correct run</div>
      {run(False)}
      <div class="runlabel">Same run, one planted mistake: <span class="reply">the eligibility
      check is skipped</span></div>
      {run(True)}
    </div>
    <table style="margin-top:14px">
      <tr><th></th><th>correct run</th><th>with the mistake</th><th>Δ</th></tr>
      <tr><td class="reply">Judge that reads the reply only</td><td>flagged</td><td>flagged</td><td class="reply">0</td></tr>
      <tr><td class="step">Judge that checks each step</td><td>passed</td><td>flagged</td><td class="step">1</td></tr>
    </table>
  </div>
  <p>Yellow is what a reply-only judge reads. Both runs end with the <b>same reply</b>, so this
  judge has to treat them the same: if it flags the mistake, it flags the correct run too. That
  is a guess, and we call this kind of mistake a <b>hidden mistake</b>.</p>

  <h2>How we test it</h2>
  <div class="tiles">
    <div class="tile"><b>{m["tjNTraj"]}</b><span>agent runs</span></div>
    <div class="tile"><b>{m["tjNClean"]}</b><span>correct</span></div>
    <div class="tile"><b>{m["tjNFaulty"]}</b><span>with one mistake each</span></div>
    <div class="tile"><b>6</b><span>mistake types</span></div>
  </div>
  <p>A small customer-support desk with 7 tools, and a scripted agent that always does the job
  right. We break one step and replay the run, so we know for sure which runs are wrong. No
  human labelling is needed.</p>
  <div class="panel">
    <p><b>Our score Δ: can the judge tell a run from its twin?</b></p>
    <p>Δ = how often it flags the broken run &minus; how often it flags its correct twin.
    Δ = 1 means it always tells them apart; Δ = 0 means it never can.</p>
  </div>

  <h2>Many flags, but no real catches</h2>
  <div class="figure"><picture><source media="(max-width: 600px)"
  srcset="figures/by_type_narrow.svg"><img src="figures/by_type.svg" alt="For four mistake types that leave the
  reply unchanged, the reply-only judge flags broken runs and their correct twins equally (delta
  0.00), while the step-by-step judge reaches delta 1.00."></picture></div>
  <p class="caption">Ring = correct twin, dot = broken run. On hidden mistakes (yellow band), the
  <span class="reply">reply-only judge</span> flags {m["tjOutcomeReplySameRecallMinPct"]} to
  {m["tjOutcomeReplySameRecallMaxPct"]} of them, but flags their twins just as often: the dot sits
  on the ring. The <span class="step">step-by-step judge</span> catches every one.</p>

  <h2>Checking each step works</h2>
  <div class="tiles">
    <div class="tile"><b>{c["hidden_caught"]}/{c["hidden_n"]}</b><span>hidden mistakes caught</span></div>
    <div class="tile"><b>{c["step_clean_flags"]}/{m["tjNClean"]}</b><span>false alarms on correct runs</span></div>
  </div>
  <p>Both judges are the same AI model (Qwen2.5, 14B). Only what it reads and what we ask it
  differ.</p>

  <h2>What matters: what it sees, or what we ask?</h2>
  <div class="panel scroll">
    <table class="grid2">
      <tr><th></th><th>asked to judge the reply</th><th>asked to check each step</th></tr>
      <tr><td><b>sees the reply only</b></td>
        <td class="num">{m["tjAblAPairReplySame"]}<small>false alarms {m["tjAblAFAK"]}/{m["tjAblAFAN"]}</small></td>
        <td class="num">{m["tjAblBPairReplySame"]}<small>false alarms {m["tjAblBFAK"]}/{m["tjAblBFAN"]}</small></td></tr>
      <tr><td><b>sees every step</b></td>
        <td class="num">{m["tjAblCPairReplySame"]}<small>false alarms {m["tjAblCFAK"]}/{m["tjAblCFAN"]}</small></td>
        <td class="num">{m["tjAblDPairReplySame"]}<small>false alarms {m["tjAblDFAK"]}/{m["tjAblDFAN"]}</small></td></tr>
    </table>
  </div>
  <p class="caption">Δ on hidden mistakes (1 is perfect), same model, in a new test with
  {m["tjAblAFAN"]} correct runs. Showing the steps is a must; asking the judge to check them does
  the rest.</p>

  <h2>But it rarely checks the reply</h2>
  <div class="panel">
    <p><b>The reply ends:</b> &ldquo;&hellip; I have also cancelled the subscription linked to
    this order.&rdquo; <span style="color:var(--muted)">There is no subscription.</span></p>
    <div class="quote"><b>The judge that checks each step:</b> &ldquo;The agent's trajectory
    follows the procedure correctly, step by step&hellip;&rdquo;<br>
    Verdict: <b>clean</b>, confidence {m["tjCaseConf"]}</div>
  </div>
  <p>We added 4 kinds of made-up sentence to the reply. The step-by-step judge catches one of
  them (a voucher apology) {m["tjStepQUnsupVoucherK"]} of {m["tjStepQUnsupVoucherN"]} times. The
  other three it catches {m["tjStepQUnsupOtherK"]} time in {m["tjStepQUnsupOtherN"]}.</p>

  <h2>Recall can fool you</h2>
  <div class="figure"><picture><source media="(max-width: 600px)"
  srcset="figures/judges_narrow.svg"><img src="figures/judges.svg" alt="Recall and delta for five judges. The
  small 8B judge has the highest recall, 0.99, and a delta of -0.01."></picture></div>
  <p class="caption">Grey is how much a judge flags, black is how much it really catches. The
  small judge flags {m["tjStepLFlags"]} of {m["tjNTraj"]} runs: the best recall and no real
  catches. Asking the same judge 3 times costs 3&times; as much and does not help.</p>

  <h2>Take home</h2>
  <ol>
    <li><b>Test each broken run against its correct twin</b> (Δ), and report recall next to
    it.</li>
    <li><b>Report hidden mistakes separately.</b> They change nothing in the reply.</li>
    <li><b>Test the reply on its own.</b> A step-by-step judge can miss false claims there.</li>
  </ol>

  <h2>Try it</h2>
  <pre><code>pip install trajectory-judge
docker run --rm ghcr.io/mohammadi-hadi/trajectory-judge   # mock judge, no model needed</code></pre>
  <p>Every number on this page and in the paper can be rebuilt offline from the committed
  verdicts with <code>make report</code> and <code>make numbers</code>. See the
  <a href="{REPO}#reproducing-the-paper">repository</a>.</p>

  <h2>Cite</h2>
  <pre><code>{html.escape(bibtex)}</code></pre>

  <h2>About me</h2>
  <p>I'm Hadi Mohammadi. I did my PhD at Utrecht University on explainable NLP and on how far
  we can trust LLMs as annotators and judges. I build tests that show where AI judges and agents
  fail. I'm open to research and applied AI roles.</p>
  <p class="about"><a href="{CV}">CV (PDF)</a> <a href="https://mohammadi.cv">mohammadi.cv</a>
  <a href="https://github.com/mohammadi-hadi">GitHub</a></p>

  <footer>
    <a href="{REPO}">Source &amp; methodology</a> &middot;
    <a href="{ARXIV}">Paper</a> &middot; MIT licensed
  </footer>
</main>
</body>
</html>
"""


def main() -> None:
    (DOCS / "figures").mkdir(parents=True, exist_ok=True)
    setup_fonts()
    trajectories, verdicts, ci = mf.load()
    paired = json.loads((ROOT / "analysis" / "paired.json").read_text())
    m = macros()
    for narrow in (False, True):
        fig_by_type(trajectories, verdicts, narrow)
        fig_judges(ci, paired, narrow)
    social_card(m)
    counts = hidden_counts(trajectories, verdicts, paired)
    (DOCS / "index.html").write_text(page(m, counts, (DOCS / "poster.pdf").exists()))
    print(f"wrote {DOCS / 'index.html'}")


if __name__ == "__main__":
    main()
