"""interp_probe.py — sanity: load both target checkpoints, confirm they reproduce behavior."""
import torch
import expF_unified as F
import expG_controller as G

base = 10
print("device:", F.DEVICE)

# --- unified 4-op model ---
m = F.UnifiedMealy(base=base, state_dim=8, hidden=96)
m.load_state_dict(torch.load("runs/expF_unified_dynamic.pt", map_location=F.DEVICE)); m.to(F.DEVICE)
print("unified params:", sum(p.numel() for p in m.parameters()))
for op, a, b in [("add", 47, 58), ("sub", 803, 67), ("mul", 1234, 7), ("div", 12345, 5), ("div", 12345, 2)]:
    got = F.predict_fn(m, op, base)(a, b)
    exp = F.cd.exact_result(a, b, op)
    print(f"  {op} {a} {b} -> got {got} exp {exp} {'OK' if got==exp else 'XX'}")

# --- GRU controller ---
c = G.Controller(hidden=64)
c.load_state_dict(torch.load("runs/expG_controller.pt", map_location=G.DEVICE)); c.to(G.DEVICE)
print("controller params:", sum(p.numel() for p in c.parameters()))
for op, A, B, D in [("mul", 47, 83, 0), ("div", 1234, 0, 7), ("div", 9999, 0, 3), ("mul", 123456, 789, 0)]:
    got, steps, halted = G.controller_run(c, op, A, B, D, base)
    exp = A * B if op == "mul" else A // D
    print(f"  {op} A={A} B={B} D={D} -> got {got} exp {exp} steps {steps} halted {halted} {'OK' if got==exp and halted else 'XX'}")
print("PROBE DONE")
