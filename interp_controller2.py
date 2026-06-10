"""
interp_controller2.py — nail two things on the GRU controller:
  Q1  the OP-ROUTING asymmetry (mul latches op-identity, div must re-read it). What does div
      DO when the op bit is removed, and at which step does removal stop mattering?
  Q2  a fully-legible control FSM: key each transition by the flag the BRANCH actually reads
      (the destination step's obs), so 'COMBINE branches on ge' and 'INC_J branches on done'
      become explicit.
"""
import random
import numpy as np
import torch
import expG_controller as G

base = 10; DEV = G.DEVICE; INSTRS = G.INSTRS
model = G.Controller(hidden=64)
model.load_state_dict(torch.load("runs/expG_controller.pt", map_location=DEV)); model.to(DEV); model.eval()


@torch.no_grad()
def step_gru(obs_vec, h):
    obs = torch.tensor([obs_vec], dtype=torch.float32, device=DEV).unsqueeze(0)
    logits, h = model(obs, h)
    return INSTRS[int(logits[0, -1].argmax())], h


@torch.no_grad()
def record_run(op, A, B, D, obs_override=None, cap=None):
    vm = G.VM(op, A, B, D, base)
    if cap is None: cap = 6 * vm.N + vm.N * base + 30
    h = None; rec = []; steps = 0
    while not vm.halted and steps < cap:
        o = vm.obs()
        if obs_override is not None: o = obs_override(o, steps, vm)
        instr, h = step_gru(o, h); rec.append((list(o), instr)); vm.execute(instr); steps += 1
    return vm, rec


print("=" * 78)
print("Q1. OP-ROUTING ASYMMETRY")
print("=" * 78)
# what does div emit when op is zeroed after step 1?
def zero_op_after(k):
    def f(o, step, vm):
        return [0.0, 0.0, o[2], o[3]] if step >= k else o
    return f


vm, rec = record_run("div", 1234, 0, 7, obs_override=zero_op_after(1))
print(f"  div 1234/7 with op-bits zeroed after step1 -> emits:")
print("    " + " ".join(instr for _, instr in rec) + f"   (answer {vm.answer()}, should be 176)")
print("    => with op_div removed it falls into the MUL program (MULDIGIT/SHL/ADD_ACC), i.e.")
print("       op_div=1 must be RE-ASSERTED each step to hold the division control flow.")

# k-sweep: zero op after step k; does the op survive?
print("\n  step at which op-removal stops/keeps mattering (zero op AFTER step k):")
for op, A, B, D in [("mul", 47, 83, 0), ("div", 1234, 0, 7)]:
    exp = A * B if op == "mul" else A // D
    row = []
    for k in [0, 1, 2, 3, 5, 8, 100]:
        vm, _ = record_run(op, A, B, D, obs_override=zero_op_after(k))
        row.append(f"k={k}:{'OK' if (vm.answer()==exp and vm.halted) else 'X'}")
    print(f"    {op}: " + "  ".join(row))
print("    (mul OK even at k=0 -> op never needed once the rigid cycle starts = fully latched;")
print("     div needs op_div essentially always -> re-read each step.)")

# is it specifically op_div=1 that div needs, or just 'not mul'? feed op_mul=1 mid-div:
def flip_to_mul_after(k):
    def f(o, step, vm):
        return [1.0, 0.0, o[2], o[3]] if step >= k else o
    return f


vm, _ = record_run("div", 1234, 0, 7, obs_override=flip_to_mul_after(3))
print(f"  div 1234/7 but obs forced to op_mul after step3 -> answer {vm.answer()} (breaks division).")

print("\n" + "=" * 78)
print("Q2. LEGIBLE CONTROL FSM — transitions keyed by the DESTINATION obs (the flag the branch")
print("    reads). ge=VAL>=D, done=J>=N.")
print("=" * 78)
data = {"mul": [], "div": []}
rng = random.Random(0)
for _ in range(150):
    for op in ("mul", "div"):
        w = rng.choice([1, 2, 3, 4]); A = rng.randint(0, base ** w - 1)
        B = rng.randint(0, base ** w - 1) if op == "mul" else 0
        D = 0 if op == "mul" else rng.randint(1, base - 1)
        vm, rec = record_run(op, A, B, D); data[op].append(rec)

for op in ("mul", "div"):
    trans = {}  # (src_instr, dst_obs) -> set(dst_instr)
    for rec in data[op]:
        for i in range(len(rec) - 1):
            src = rec[i][1]; dst_obs = tuple(int(x) for x in rec[i + 1][0]); dst = rec[i + 1][1]
            trans.setdefault((src, dst_obs), set()).add(dst)
    print(f"  [{op}] control FSM:")
    seen = set()
    for (src, dobs), dsts in sorted(trans.items(), key=lambda kv: (INSTRS.index(kv[0][0]), kv[0][1])):
        ge, done = dobs[2], dobs[3]
        cond = f"ge={ge},done={done}"
        print(f"      {src:9s} --[{cond}]--> {sorted(dsts)}")
print("\nINTERP_CONTROLLER2 DONE")
