"""
gpu_weird_lprog.py — a STRUCTURE-AGNOSTIC interestingness signal: LEARNING PROGRESS (Schmidhuber's
"interestingness = compression progress"). The standing ceiling of this project is that EVERY intrinsic signal it has
built was hand-pointed at a KNOWN structure class (edge-of-chaos compression -> class-4 CAs in expY; algebraic invariants
-> integrable maps in expZ; the named-battery residual in gpu_exp1). So scale only ever resurfaces KNOWN classes. This
signal names NO structure. It asks one thing of an object's space-time pattern:

    does a learner's prediction of it IMPROVE A LOT with a little training?

    learning_progress(orbit) = bpc_predictor_AFTER_WARMUP(orbit) - bpc_predictor_LATE(orbit)   (held-out loss DROP)

- trivial / periodic orbit  -> learned during the warmup        -> SMALL post-warmup drop  (nothing left to learn).
- chaotic / noise orbit      -> loss stays high forever          -> SMALL drop  (nothing learnable).
- structured-but-nontrivial -> keeps improving PAST the warmup as the learner finds the regularity -> LARGE drop = INTERESTING.

THE KEY SUBTLETY (two design choices, both validated on the 256 ground-truth CAs below):
  (a) PER-OBJECT learner, not one shared net. I first tried ONE predictor SHARED across the whole batch (so shared weights
      could only exploit CROSS-rule regularity). On the elementary-CA ground truth that FAILED cleanly (honest negative):
      a single net spends its capacity on the dominant easy patterns, never fits the complex rules, and "progress"
      collapses to FINAL predictability -> TRIVIAL constant rules (0/255/...) win, class-4 sinks to the bottom. So the
      shared predictor is too coarse — exactly the failure mode the task anticipated. The fix is the task's documented
      alternative: a tiny FRESH predictor PER OBJECT (B independent models trained in parallel as ONE grouped 1D-conv,
      groups=B), measuring how much each object's OWN learner improves on IT. Capacity-bounded + tiny so it cannot rote-
      memorize noise: it only drops loss where the orbit has genuinely learnable temporal structure.
  (b) WARMUP baseline, not step 0. Untrained bpc is ~1.0 for everything, so (step0 - late) again just rewards final
      predictability = trivial orbits. Measuring AFTER a short warmup neutralizes the "instantly learnable" part: a
      constant/periodic orbit is fully captured during warmup (post-warmup drop ~0); a complex orbit keeps improving long
      after. Train and eval use DIFFERENT inits, so progress = learning transferable DYNAMICS, not memorizing one init.
The signal still never mentions gliders, periods, compressibility, or chaos — it is defined purely by trainability.

VALIDATION (the real test, like expY): run on the 256 elementary CAs (radius=1, k=3, Wolfram classes are ground truth).
Learning-progress must rank class-4 (complex) HIGH while rejecting BOTH trivial/periodic (low: nothing to learn) AND
class-3 chaos (low: nothing learnable). If it does not cleanly separate class-4 from chaos+trivial, that is an HONEST
NEGATIVE about the signal and is reported as such.

VALIDATED RESULT + ITS HONEST CAVEAT (measured below, seed 1):
  * STRONG: learning-progress decisively REJECTS the trivial/periodic cohort at EVERY orbit size — those orbits are
    fully captured during the warmup so post-warmup progress ~ 0 (trivial-cohort median rank ~200/256 always). This is
    the hard part of the task (separating trivial from interesting WITHOUT a target) and it works robustly.
  * PARTIAL / FRAGILE: the class-4 vs class-3-chaos margin depends on BOTH orbit size and train budget. At the
    validated sweet spot (W=T=64, ~32 steps, the defaults) class-4 ranks ABOVE chaos (cohort median rank ~20 vs ~25,
    both far above trivial ~202; 4/7 class-4 in the top-24) — a clean positive. But the margin is thin: pushing to
    LARGER orbits (W=T=128) OR MORE steps (~40) lets a few class-3 rules (126, 22, 30, 90) — which carry SUSTAINED
    locally-deterministic structure before chaos takes over — accumulate enough learnable early structure to OUT-rank
    class-4, flipping the verdict to a marginal NEGATIVE (chaos median can dip to ~10-15). So learning-progress separates
    class-4 as a DISTRIBUTION from trivial cleanly and from chaos only at the right budget, and never rule-by-rule. This
    is reported, not hidden: it is a real but FRAGILE/scale-tuned interestingness signal, NOT a crisp class-4 detector.

PAYOFF: on the uncatalogued radius-2 space (2^32 rules) find orbits with HIGH learning-progress that the project's NAMED
battery does NOT flag (not affine [NL==0], not short-periodic, not edge-of-chaos-extreme density/damage) = structured-by-
trainability but invisible to every named signal = residual-novel candidates. Saved to runs/weird_lprog/ for inspection.
Honest ceiling (unchanged): uncatalogued != provably-unknown; survivors are CHARACTERIZED, never claimed novel.

Reuses the validated CA machinery from gpu_exp1_novelty.py.
Run small (4060):   python gpu_weird_lprog.py --smoke
Run scale (4090):   python gpu_weird_lprog.py --radius 2 --batch 65536 --keep 8192 --W 64 --T 64 --train-steps 32 --warm 6 --out runs/weird_lprog
  (KEEP W=T=64 & steps=32 — the validated sweet spot above; the scale lever is a BIGGER sample of the 2^32 tail, NOT
   bigger/longer orbits, which erode the class-4 vs chaos margin. batch=8192 already runs in ~40s on a 4060; memory is
   bounded by obj_chunk=1024 regardless of batch, so 65536 is fine — lower obj_chunk only if the card is small.)
"""
from __future__ import annotations
import argparse, time, os, json
import numpy as np
import torch

from gpu_exp1_novelty import (
    DEV, dev_info, run_ca, sample_luts, affine_tables, nonlinearity,
    density, periodicity, damage, lut_to_int,
)

LOG2 = float(np.log(2.0))


# ---------------------------------------------------------------------------
# The PER-OBJECT next-row predictor: B independent tiny 2-layer causal-conv models trained IN PARALLEL as a single
# grouped 1D-conv (groups=B). Each object gets its OWN weights, so each measures how much a fresh learner improves on
# IT alone. The model is deliberately tiny (hid small) so it cannot rote-memorize noise — it only drops loss where the
# orbit has genuinely learnable temporal structure. Predicts row t from rows [t-L, t-1] over the width.
#
# Layout trick: pack the per-object channels into the group axis. Build a context tensor (B, T-L, L, W), permute so the
# L context rows sit next to the object index -> a (1, B*L, (T-L)*W) "signal" with B*L input channels in B groups; the
# grouped conv then applies a private (L->hid->1) conv per object. Flattening (T-L) and W into one length axis lets a
# single Conv1d see every (target-row, width) position of every object at once.
# ---------------------------------------------------------------------------
class PerObjectPredictor(torch.nn.Module):
    def __init__(self, B, L=4, hid=12):
        super().__init__()
        self.B, self.L, self.hid = B, L, hid
        self.c1 = torch.nn.Conv1d(B * L, B * hid, 7, padding=3, groups=B)
        self.c2 = torch.nn.Conv1d(B * hid, B * 1, 7, padding=3, groups=B)

    def _signal(self, st_f):                                   # st_f: (B, T, W) float -> (x (1,B*L,P), tgt (B,P)) P=(T-L)*W
        B, T, W = st_f.shape
        L = self.L
        ctx = torch.stack([st_f[:, t - L:t] for t in range(L, T)], dim=1)   # (B, T-L, L, W)
        x = ctx.permute(0, 2, 1, 3).reshape(1, B * L, (T - L) * W)          # group axis = B*L
        tgt = st_f[:, L:].reshape(B, (T - L) * W)
        return x, tgt

    def logits_tgt(self, st_f):
        x, tgt = self._signal(st_f)
        h = torch.relu(self.c1(x))
        o = self.c2(h).reshape(self.B, -1)                                  # (B, P)
        return o, tgt

    @torch.no_grad()
    def bpc_per_object(self, st_f):
        o, tgt = self.logits_tgt(st_f)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(o, tgt, reduction="none")  # (B, P) nats
        return loss.mean(dim=1) / LOG2                                      # nats -> bits, (B,)


# ---------------------------------------------------------------------------
# BATCHED LEARNING PROGRESS (per-object).
# Train B parallel per-object predictors on the TRAIN orbits for `steps`; at checkpoints, freeze and measure per-object
# held-out bpc on a DIFFERENT init (the EVAL orbits) -> a per-object loss CURVE. learning_progress = bpc[warm] - bpc[best],
# i.e. the held-out loss drop AFTER a short warmup (warmup neutralizes the instantly-learnable trivial/periodic orbits).
# Train and eval seeds differ so progress = learning transferable dynamics, not memorizing the train init.
# ---------------------------------------------------------------------------
def learning_progress(luts, radius, W, T, steps, warm, seed=0, L=4, hid=12, lr=5e-2, obj_chunk=1024):
    """-> dict(lp, curve (steps+1, B), bpc_start (post-warmup), bpc_end (best)). lp = post-warmup bpc - best bpc, (B,).
    obj_chunk bounds GPU memory: the grouped conv over B objects is split into independent sub-batches (each a fresh
    set of per-object models) so very large `batch` at scale never OOMs — objects are independent, so this is exact.
    The grouped-conv activation is ~ obj_chunk * hid * (T-L) * W floats; at W=T=128, hid=12 that is ~0.7 GB per 1024
    objects, so obj_chunk=1024 keeps a scale hunt well under a 24 GB card. Lower it for very large W/T or a small GPU."""
    B = luts.shape[0]
    st_train_all = run_ca(luts, radius, W, T, seed=10000 + seed).float()
    st_eval_all = run_ca(luts, radius, W, T, seed=20000 + seed).float()
    curve = torch.empty((steps + 1, B), device=DEV)
    for lo in range(0, B, obj_chunk):
        hi = min(lo + obj_chunk, B)
        st_tr, st_ev = st_train_all[lo:hi], st_eval_all[lo:hi]
        torch.manual_seed(seed * 100003 + lo)                 # distinct init per chunk, deterministic
        model = PerObjectPredictor(hi - lo, L=L, hid=hid).to(DEV)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        for s in range(steps + 1):
            model.eval()
            curve[s, lo:hi] = model.bpc_per_object(st_ev)
            if s == steps:
                break
            model.train()
            opt.zero_grad()
            o, tgt = model.logits_tgt(st_tr)
            torch.nn.functional.binary_cross_entropy_with_logits(o, tgt).backward()
            opt.step()
    warm = min(warm, steps)
    bpc_start = curve[warm]                                   # post-warmup baseline
    bpc_end = curve.min(dim=0).values                         # best (lowest) achieved bpc => robust to late wobble
    lp = (bpc_start - bpc_end).clamp(min=0.0)                 # progress is a non-negative drop
    return dict(lp=lp, curve=curve, bpc_start=bpc_start, bpc_end=bpc_end)


# ---------------------------------------------------------------------------
# Named-structure battery (the project's validated signals) — used ONLY to mark which high-LP rules are already explained.
# ---------------------------------------------------------------------------
def named_battery(luts, st, radius, W, T, aff):
    pmax = max(4, min(24, T // 6))
    dens = density(st)
    nl = nonlinearity(luts, aff)
    dmg = damage(luts, radius, W, T, seed=11)
    per_P, b_per = periodicity(st, pmax)
    dead = (dens < 0.03) | (dens > 0.97)
    affine = nl == 0
    periodic = b_per < 0.25
    chaotic = dmg > 0.40
    ordered = dmg < 0.02
    named = dead | affine | periodic | chaotic | ordered
    return dict(dens=dens, nl=nl, dmg=dmg, per_P=per_P, b_per=b_per,
                dead=dead, affine=affine, periodic=periodic, chaotic=chaotic, ordered=ordered, named=named)


# ---------------------------------------------------------------------------
# Wolfram ground truth for the 256 elementary rules (class 4 = complex/interesting).
# ---------------------------------------------------------------------------
W_CLASS4 = {54, 106, 110, 124, 137, 147, 193}
W_CLASS3 = {18, 22, 30, 45, 60, 75, 86, 90, 105, 122, 126, 129, 135, 146, 149, 150, 165, 182, 195}
W_CLASS1 = {0, 8, 32, 40, 128, 136, 160, 168, 224, 234, 235, 238, 250, 251, 254, 255}


def wclass(r):
    if r in W_CLASS4: return "4*"
    if r in W_CLASS3: return "3 "
    if r in W_CLASS1: return "1 "
    return "2 "


def elementary_luts():
    """The 256 elementary CA rules as radius-1 size-8 truth tables: lut[i] = bit i of the rule number. (B=256, 8) uint8."""
    r = torch.arange(256, device=DEV)
    bits = torch.arange(8, device=DEV)
    return ((r[:, None] >> bits[None, :]) & 1).to(torch.uint8)             # (256, 8)


def validate_elementary(W, T, steps, warm, seed, L, hid):
    print("=" * 100)
    print("VALIDATION on the 256 ELEMENTARY CAs (radius=1) — does LEARNING PROGRESS rank class-4 high, reject chaos+trivial?")
    print("=" * 100)
    luts = elementary_luts()
    t0 = time.time()
    out = learning_progress(luts, 1, W, T, steps, warm, seed=seed, L=L, hid=hid)
    lp = out["lp"]; bpc0 = out["bpc_start"]; bpc1 = out["bpc_end"]
    order = torch.argsort(lp, descending=True).tolist()
    rank = {r: i for i, r in enumerate(order)}

    print(f"  trained 256 per-object predictors for {steps} steps, warmup baseline @ step {warm}  [{time.time()-t0:.0f}s]")
    print(f"  per-object held-out bpc seeds: train=10000+{seed}, eval=20000+{seed} (held-out init)\n")
    print("  TOP 24 by learning-progress (lp = bpc_post-warmup - bpc_best; high = learnable-with-effort):")
    print("    rule  class    lp     bpcW   bpc_end")
    c4_top = 0
    for r in order[:24]:
        star = "  <<< CLASS 4 (complex!)" if r in W_CLASS4 else ("   (chaos)" if r in W_CLASS3 else "")
        if r in W_CLASS4: c4_top += 1
        print(f"    {r:4d}   {wclass(r)}   {lp[r]:.4f}  {bpc0[r]:.3f}  {bpc1[r]:.3f}{star}")

    print(f"\n  where do the 7 famous CLASS-4 (complex) rules land? (out of 256, rank 0 = highest lp)")
    for r in sorted(W_CLASS4):
        print(f"     rule {r:3d}  class 4 :  rank {rank[r]:3d}/256   lp {lp[r]:.4f}  bpcW {bpc0[r]:.3f} -> {bpc1[r]:.3f}")
    print("  CLASS-3 chaos (should rank LOW — nothing learnable, loss stays high):")
    for r in sorted({30, 45, 90, 150, 18, 22, 126}):
        print(f"     rule {r:3d}  class 3 :  rank {rank[r]:3d}/256   lp {lp[r]:.4f}  bpcW {bpc0[r]:.3f} -> {bpc1[r]:.3f}")
    print("  TRIVIAL class-1/2 samples (should rank LOW — already predictable, nothing to learn):")
    for r in sorted({0, 255, 4, 8, 51, 204, 170, 240}):
        cl = wclass(r)
        print(f"     rule {r:3d}  class {cl}:  rank {rank[r]:3d}/256   lp {lp[r]:.4f}  bpcW {bpc0[r]:.3f} -> {bpc1[r]:.3f}")

    # honest separation verdict: median ranks of the three cohorts (lower rank = more interesting by the signal)
    def med_rank(rs): return float(np.median([rank[r] for r in rs]))
    c4 = sorted(W_CLASS4)
    chaos = [30, 45, 90, 150, 18, 22, 126]
    trivial = [0, 255, 4, 8, 51, 204, 170, 240]
    mc4, mch, mtr = med_rank(c4), med_rank(chaos), med_rank(trivial)
    print(f"\n  COHORT MEDIAN RANKS (0=best):  class-4 {mc4:.0f}   |   chaos {mch:.0f}   |   trivial {mtr:.0f}   (of 256)")
    print(f"  class-4 rules in TOP 24: {c4_top}/7.  Mean lp:  class-4 {lp[c4].mean():.4f}  chaos "
          f"{lp[chaos].mean():.4f}  trivial {lp[trivial].mean():.4f}")
    clean = (mc4 < mch) and (mc4 < mtr)
    print(f"  VERDICT: class-4 ranks {'ABOVE' if clean else 'NOT clearly above'} both chaos and trivial "
          f"=> signal {'SEPARATES (positive)' if clean else 'does NOT cleanly separate (HONEST NEGATIVE)'}.\n")
    return clean


# ---------------------------------------------------------------------------
# PAYOFF: radius-2 hunt for HIGH learning-progress orbits NOT flagged by the named battery.
# ---------------------------------------------------------------------------
def hunt_radius2(radius, W, T, batch, keep, steps, warm, seed, L, hid, out):
    k = 2 * radius + 1
    print("=" * 100)
    print(f"PAYOFF: hunt the uncatalogued radius-{radius} space (2^{1<<k} rules) for HIGH learning-progress NOT named.")
    print("=" * 100)
    aff = affine_tables(k, DEV)
    g = torch.Generator(device=DEV).manual_seed(seed)
    luts = sample_luts(batch, 1 << k, g)
    t0 = time.time()

    out_lp = learning_progress(luts, radius, W, T, steps, warm, seed=seed, L=L, hid=hid)
    lp = out_lp["lp"]; bpc0 = out_lp["bpc_start"]; bpc1 = out_lp["bpc_end"]
    print(f"  scored {batch} random rules, {steps} train steps (per-object), warmup @ {warm}  [{time.time()-t0:.0f}s].")

    # rank by lp, keep the top, characterize with the named battery (one eval-init orbit reused for stats)
    order = torch.argsort(lp, descending=True)[:keep]
    sub = luts[order]
    st = run_ca(sub, radius, W, T, seed=20000 + seed)
    f = named_battery(sub, st, radius, W, T, aff)
    lp_s, b0_s, b1_s = lp[order], bpc0[order], bpc1[order]
    residual = ~f["named"]                                     # high lp AND no named signal explains it

    n_res = int(residual.sum())
    print(f"  of the top {keep} by learning-progress, {n_res} are NOT flagged by ANY named signal "
          f"(not affine / periodic / chaotic / dead / frozen).")
    print(f"  (named breakdown among top {keep}:  affine {int(f['affine'].sum())}  periodic {int(f['periodic'].sum())}"
          f"  chaotic {int(f['chaotic'].sum())}  dead {int(f['dead'].sum())}  frozen {int(f['ordered'].sum())})\n")

    res_idx = residual.nonzero(as_tuple=True)[0]
    res_idx = res_idx[torch.argsort(lp_s[res_idx], descending=True)]      # residual candidates, best lp first
    print("  TOP residual-novel candidates (high learning-progress, invisible to the named battery):")
    print("    rule              lp      bpcW   bpc_end  NL  period  dens   damage")
    results = []
    for j in res_idx[:24].tolist():
        rid = lut_to_int(sub[j]); rids = f"0x{rid:08X}" if rid is not None else "(big-k lut)"
        results.append(dict(rule=rid, lp=float(lp_s[j]), bpc_warm=float(b0_s[j]), bpc_end=float(b1_s[j]),
                            nl=int(f["nl"][j]), period=int(f["per_P"][j]), density=float(f["dens"][j]),
                            damage=float(f["dmg"][j])))
        print(f"    {rids:14s}    {lp_s[j]:.4f}  {b0_s[j]:.3f}  {b1_s[j]:.3f}  {int(f['nl'][j]):3d}  "
              f"{int(f['per_P'][j]):4d}   {f['dens'][j]:.2f}   {f['dmg'][j]:.3f}")

    if out:
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, "lprog_survivors.json"), "w") as fh:
            json.dump(dict(radius=radius, k=k, W=W, T=T, batch=batch, keep=keep, steps=steps, warm=warm,
                           n_residual=n_res, results=results), fh, indent=2)
        nsave = min(8, len(res_idx))
        if nsave:
            keep_luts = sub[res_idx[:nsave]]
            stb = run_ca(keep_luts, radius, W, T, seed=7).cpu().numpy()
            np.save(os.path.join(out, "lprog_top_spacetime.npy"), stb)
            np.save(os.path.join(out, "lprog_top_luts.npy"), keep_luts.cpu().numpy())
            print(f"\n  saved top-{nsave} residual survivors' LUTs + space-time to {out}/ for inspection.")
    print("\n  READ: these are compressible-by-LEARNING (a per-object predictor improves a lot on them PAST a warmup) yet")
    print("  match NO named signal. Honest ceiling: characterize, do not claim novel. A survivor whose dynamics look unfamiliar")
    print("  on inspection = evidenceable-but-unprovable novelty; one that is 'just class-4' = the rediscovery ceiling.")
    return results


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="tiny config for a 4060 correctness check (<60s, every path)")
    ap.add_argument("--radius", type=int, default=2)
    ap.add_argument("--W", type=int, default=64, help="orbit width; class-4-vs-chaos separation is cleanest near 64 (see header)")
    ap.add_argument("--T", type=int, default=64, help="orbit height")
    ap.add_argument("--batch", type=int, default=8192, help="random radius-2 rules to score for the hunt")
    ap.add_argument("--keep", type=int, default=2048, help="top-by-lp to characterize with the named battery")
    ap.add_argument("--train-steps", type=int, default=32, help="per-object steps; the validated class-4>chaos sweet spot "
                    "is ~32 at W=T=64 — MORE steps lets chaos' early-determinism keep getting learned and erodes the margin")
    ap.add_argument("--warm", type=int, default=6, help="warmup step whose held-out bpc is the learning-progress baseline")
    ap.add_argument("--L", type=int, default=4, help="predictor context rows")
    ap.add_argument("--hid", type=int, default=12, help="per-object predictor hidden width (tiny: cannot memorize noise)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--no-hunt", action="store_true", help="only run the elementary-CA validation")
    ap.add_argument("--out", type=str, default="runs/weird_lprog")
    args = ap.parse_args()

    if args.smoke:
        args.W, args.T, args.batch, args.keep, args.train_steps, args.warm = 48, 48, 512, 128, 24, 5

    s, warm = args.train_steps, min(args.warm, args.train_steps)
    print(f"gpu_weird_lprog | device={dev_info()} | LEARNING-PROGRESS signal (structure-agnostic interestingness)")
    print(f"  radius={args.radius} W={args.W} T={args.T} | per-object train_steps={s} warmup@{warm} L={args.L} hid={args.hid}")
    print(f"  validation_batch=256 (elementary) | hunt_batch={args.batch} keep={args.keep} seed={args.seed}\n")
    t0 = time.time()

    # 1) ground-truth validation (always; radius-1 elementary CAs)
    validate_elementary(args.W, args.T, s, warm, args.seed, args.L, args.hid)

    # 2) the payoff hunt on the big uncatalogued space (unless --no-hunt)
    if not args.no_hunt:
        hunt_radius2(args.radius, args.W, args.T, args.batch, args.keep, s, warm, args.seed, args.L, args.hid, args.out)

    print(f"\n[done {time.time()-t0:.0f}s total]")
