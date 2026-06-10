"""
interp_controller.py — DEEP interpretability of the GRU controller (runs/expG_controller.pt).

The controller emits a multi-step PROGRAM (instruction per step) over a register VM, seeing only
a 4-dim observation [op_mul, op_div, ge, done]. That obs is ~CONSTANT across the distinct
instructions of a cycle, so the controller MUST hold the program phase in its 64-dim hidden state.
What is that hidden state actually doing?

Sections:
  P1  PROGRAM COUNTER: feed a CONSTANT mul obs and watch the hidden state cycle through the
      5-instruction body on its own (the recurrence is the program counter, not the obs).
  P2  INNER LOOP: is the div SUB_D loop a flag-gated FIXED POINT (count-invariant -> generalizes
      to any quotient digit / length) or a counter? Hold ge=1 and test for a fixed point; then
      flip ge=0 and confirm it exits to STOREQ.
  P3  FSM EXTRACTION: label hidden states by emitted instruction, verify tight per-phase clusters,
      print the control FSM (states + obs-gated transitions) for mul and div.
  P4  OP ROUTING: is op latched in the hidden state or re-read each step? Corrupt the op bits in
      obs after step 1 and see whether the program still runs correctly.
  P5  LENGTH-GEN: show the hidden trajectory is the SAME cycle at width 2 and width 8.
"""
import random
import numpy as np
import torch
import expG_controller as G

base = G.base if hasattr(G, "base") else 10
DEV = G.DEVICE
INSTRS = G.INSTRS
model = G.Controller(hidden=64)
model.load_state_dict(torch.load("runs/expG_controller.pt", map_location=DEV)); model.to(DEV); model.eval()


@torch.no_grad()
def step_gru(obs_vec, h):
    obs = torch.tensor([obs_vec], dtype=torch.float32, device=DEV).unsqueeze(0)  # (1,1,4)
    logits, h = model(obs, h)
    instr = INSTRS[int(logits[0, -1].argmax())]
    return instr, h


@torch.no_grad()
def record_run(op, A, B, D, obs_override=None, cap=None):
    """Run the controller on a real problem, recording (obs, h_post, instr) per step."""
    vm = G.VM(op, A, B, D, base)
    if cap is None: cap = 6 * vm.N + vm.N * base + 30
    h = None; rec = []; steps = 0
    while not vm.halted and steps < cap:
        o = vm.obs()
        if obs_override is not None: o = obs_override(o, steps, vm)
        instr, h = step_gru(o, h)
        rec.append((list(o), h.squeeze().cpu().numpy().copy(), instr))
        vm.execute(instr); steps += 1
    return vm, rec


# ---------------------------------------------------------------- P1
print("=" * 78)
print("P1. PROGRAM COUNTER — feed a CONSTANT mul obs [op_mul=1, op_div=0, ge=0, done=0] and")
print("    watch the controller cycle through the 5-instruction body autonomously.")
print("    (obs never changes -> any structure in the instruction stream is a hidden-state PC.)")
print("=" * 78)
h = None; seq = []
for _ in range(22):
    instr, h = step_gru([1.0, 0.0, 0.0, 0.0], h); seq.append(instr)
print("    emitted:", " ".join(seq))
# detect period
body = seq[:15]
print(f"    -> period-5 body cycle: {body[0:5]} repeating" if body[0:5] == body[5:10] == body[10:15]
      else f"    -> sequence (no clean period-5): {body}")

# ---------------------------------------------------------------- P2
print("\n" + "=" * 78)
print("P2. INNER SUB_D LOOP — flag-gated fixed point, or a counter?")
print("=" * 78)
# drive into the div SUB_D phase using a real problem, then probe
# build the SUB_D-phase hidden state from a real division with a big quotient digit (9/1)
vm, rec = record_run("div", 99999, 0, 1)  # every digit q=9 -> long SUB_D runs
sub_states = [h for (o, h, instr) in rec if instr == "SUB_D"]
print(f"    real div 99999/1: collected {len(sub_states)} SUB_D-phase hidden states")
# consecutive SUB_D hidden-state drift within the loop
seqd = [(o, h, instr) for (o, h, instr) in rec]
drifts = []
for i in range(1, len(seqd)):
    if seqd[i][2] == "SUB_D" and seqd[i - 1][2] == "SUB_D":
        drifts.append(np.linalg.norm(seqd[i][1] - seqd[i - 1][1]))
print(f"    consecutive-SUB_D hidden drift ||h_t - h_(t-1)||: mean {np.mean(drifts):.4f} "
      f"max {np.max(drifts):.4f}  (->0 = fixed point = count-invariant loop)")
# direct fixed-point probe: from a SUB_D hidden state, hold ge=1 for 30 steps
h0 = torch.tensor(sub_states[0], dtype=torch.float32, device=DEV).reshape(1, 1, 64)
hh = h0.clone(); instrs_held = []; norms = []
for _ in range(30):
    instr, hh = step_gru([0.0, 1.0, 1.0, 0.0], hh)
    instrs_held.append(instr); norms.append(float(torch.norm(hh - h0)))
allsub = all(x == "SUB_D" for x in instrs_held)
print(f"    hold ge=1 for 30 steps from a SUB_D state: emits SUB_D all 30? {allsub}; "
      f"||h - h0|| stays {max(norms):.4f} (bounded => stable fixed point)")
# exit test: flip ge=0 -> should emit STOREQ
instr_exit, _ = step_gru([0.0, 1.0, 0.0, 0.0], hh)
print(f"    then flip ge=0 -> emits '{instr_exit}'  (STOREQ = ge flag drives the loop exit)")

# ---------------------------------------------------------------- P3
print("\n" + "=" * 78)
print("P3. CONTROL FSM — label hidden states by emitted instruction; verify tight per-phase")
print("    clusters (within-spread << between-spread) and print obs-gated transitions.")
print("=" * 78)
data = {"mul": [], "div": []}
rng = random.Random(0)
for _ in range(120):
    for op in ("mul", "div"):
        w = rng.choice([1, 2, 3, 4]); A = rng.randint(0, base ** w - 1)
        B = rng.randint(0, base ** w - 1) if op == "mul" else 0
        D = 0 if op == "mul" else rng.randint(1, base - 1)
        vm, rec = record_run(op, A, B, D)
        data[op].append(rec)

for op in ("mul", "div"):
    H = []; lab = []; obss = []
    for rec in data[op]:
        for (o, h, instr) in rec:
            H.append(h); lab.append(instr); obss.append(tuple(int(x) for x in o))
    H = np.array(H); lab = np.array(lab)
    phases = sorted(set(lab.tolist()), key=lambda s: INSTRS.index(s))
    cent = {p: H[lab == p].mean(0) for p in phases}
    # within-phase spread vs nearest-other-centroid distance
    print(f"  [{op}] {len(H)} steps, phases (=emitted instrs): {phases}")
    for p in phases:
        within = np.linalg.norm(H[lab == p] - cent[p], axis=1).mean()
        others = [np.linalg.norm(cent[p] - cent[q]) for q in phases if q != p]
        print(f"    phase {p:9s} n={int((lab==p).sum()):4d}  within-spread {within:.3f}  "
              f"nearest-other-centroid {min(others):.3f}")
    # transition graph: (phase, obs) -> next phase
    trans = {}
    for rec in data[op]:
        for i in range(len(rec) - 1):
            o = tuple(int(x) for x in rec[i][0]); p = rec[i][2]; nxt = rec[i + 1][2]
            trans.setdefault((p, o), set()).add(nxt)
    print(f"    transitions (phase | obs[mul,div,ge,done] -> next phase):")
    for (p, o), nxts in sorted(trans.items(), key=lambda kv: INSTRS.index(kv[0][0])):
        print(f"      {p:9s} | {o} -> {sorted(nxts)}")

# ---------------------------------------------------------------- P4
print("\n" + "=" * 78)
print("P4. OP ROUTING — is op latched in the hidden state, or re-read from obs every step?")
print("=" * 78)


def zero_op_after(k):
    def f(o, step, vm):
        if step >= k: return [0.0, 0.0, o[2], o[3]]  # blank both op bits
        return o
    return f


for op, A, B, D in [("div", 1234, 0, 7), ("mul", 47, 83, 0)]:
    vm0, _ = record_run(op, A, B, D)
    vmk, _ = record_run(op, A, B, D, obs_override=zero_op_after(1))
    exp = A * B if op == "mul" else A // D
    print(f"  {op} {A}/{D if op=='div' else B}: normal got {vm0.answer()} (exp {exp}, "
          f"{'OK' if vm0.answer()==exp and vm0.halted else 'X'}); "
          f"op-bits zeroed after step1 got {vmk.answer()} halted={vmk.halted} "
          f"({'STILL CORRECT -> op is LATCHED in hidden state' if vmk.answer()==exp and vmk.halted else 'BROKE -> op re-read each step'})")

# ---------------------------------------------------------------- P5
print("\n" + "=" * 78)
print("P5. LENGTH-GEN — the hidden trajectory is the SAME cycle at width 2 and width 8.")
print("=" * 78)
# compare per-phase centroids of mul at w=2 vs w=8
def mul_centroids(w, n=60):
    H = []; lab = []
    rr = random.Random(w)
    for _ in range(n):
        A = rr.randint(0, base ** w - 1); B = rr.randint(0, base ** w - 1)
        vm, rec = record_run("mul", A, B, 0)
        for (o, h, instr) in rec: H.append(h); lab.append(instr)
    H = np.array(H); lab = np.array(lab)
    return {p: H[lab == p].mean(0) for p in set(lab.tolist())}
c2 = mul_centroids(2); c8 = mul_centroids(8)
common = sorted(set(c2) & set(c8), key=lambda s: INSTRS.index(s))
print("  per-phase centroid distance between width-2 and width-8 mul runs:")
for p in common:
    print(f"    {p:9s}: ||c_w2 - c_w8|| = {np.linalg.norm(c2[p]-c8[p]):.3f}")
print("  (~0 => identical control cycle regardless of length: the SAME finite program counter")
print("   is reused every iteration, which is exactly why it length-generalizes.)")
print("\nINTERP_CONTROLLER DONE")
