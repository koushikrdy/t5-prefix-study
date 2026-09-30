import json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 10})
runs = [json.loads(l) for l in open("results/runs.jsonl")]
C = ["correct", "wrong", "empty", "nonsense"]; L = ["correct", "wrong task", "none", "nonsense"]
G = [("A single-task", "single", "distinct", "#2f6fb5", -.27),
     ("B multi-task, disjoint labels", "multi", "distinct", "#e8703a", 0),
     ("C multi-task, collided labels", "multi", "collided", "#2bab77", .27)]
rng = np.random.default_rng(1)
fig, axs = plt.subplots(3, 1, figsize=(5, 8.2))
for ax, t, ti in zip(axs, ["cola", "sst2", "mrpc"], ["CoLA (MCC)", "SST-2 (accuracy)", "MRPC (accuracy)"]):
    for n, rg, sc, col, o in G:
        for i, c in enumerate(C):
            v = [r["eval"][t]["primary"] for r in runs if (r["regime"], r["label_scheme"], r["condition"]) == (rg, sc, c) and t in r["eval"]]
            if not v: continue
            ax.scatter(i + o + rng.uniform(-.05, .05, len(v)), v, color=col, s=30, alpha=.8, zorder=3, label=n if i == 0 else None)
            ax.hlines(np.mean(v), i + o - .11, i + o + .11, color="k", lw=2.2, zorder=4)
    ax.axhline(0 if t == "cola" else None or (.51 if t == "sst2" else .685), ls="--", color="gray", lw=1)
    if t == "sst2": ax.set_ylim(.7, .94)
    if t == "mrpc": ax.set_ylim(.68, .84)
    ax.set_xticks(range(4)); ax.set_xticklabels(L); ax.set_title(ti, fontweight="bold", loc="left")
    for s in ("top", "right"): ax.spines[s].set_visible(False)
h, l = axs[0].get_legend_handles_labels(); fig.legend(h, l, loc="upper center", frameon=False, fontsize=8.5)
fig.tight_layout(rect=(0, 0, 1, .92)); fig.savefig("fig1.png", dpi=220)

T = ["cola", "sst2", "mrpc", "(none)", "zorbex"]
fig, axs = plt.subplots(2, 1, figsize=(5, 6))
for ax, sc, ti in zip(axs, ["distinct", "collided"], ["disjoint labels", "collided labels (yes / no)"]):
    rs = [r for r in runs if r["regime"] == "multi" and r["condition"] == "correct" and r["label_scheme"] == sc]
    A = np.array([[np.mean([r["swap"][e][t]["accuracy"] for r in rs]) for t in T] for e in ["cola", "sst2", "mrpc"]])
    Rg = np.array([[np.ptp([r["swap"][e][t]["accuracy"] for r in rs]) for t in T] for e in ["cola", "sst2", "mrpc"]])
    ax.imshow(A, cmap="Blues", vmin=0, vmax=1)
    for i in range(3):
        for j in range(5):
            if Rg[i, j] > .15: ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False, hatch="///", ec="#888", lw=0))
            ax.text(j, i, f"{A[i,j]:.2f}", ha="center", va="center", fontsize=11, color="w" if A[i, j] > .55 else "k", fontweight="bold" if i == j else None)
    ax.set_xticks(range(5)); ax.set_xticklabels(["cola", "sst2", "mrpc", "none", "zorbex"]); ax.set_yticks(range(3)); ax.set_yticklabels(["CoLA", "SST-2", "MRPC"])
    ax.set_title(ti, fontweight="bold"); ax.set_xlabel("token at test time")
fig.tight_layout(); fig.savefig("fig2.png", dpi=220)

off = [r["swap"]["cola"]["(none)"]["off_label_rate"] for r in runs if r["regime"] == "multi" and r["condition"] == "correct" and r["label_scheme"] == "distinct"]
print("CoLA off-label with no token, per seed:", [round(x, 3) for x in off])