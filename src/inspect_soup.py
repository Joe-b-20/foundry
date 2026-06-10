"""inspect_soup.py — read the EMERGENT organism from a primordial-soup run, and PROVE it self-replicates.

Loads population.npy + soup.json from a soup --out dir, finds the dominant genotype(s), decodes the code (low nibble ->
op chars), and runs the dominant genotype CONCATENATED WITH A FRESH RANDOM PARTNER to test: does it copy its own bytes into
the partner half? High copy-overlap = a genuine self-replicator (life from noise), not a compression artifact.
Usage: python inspect_soup.py runs_pod/weird_soup_char
"""
import sys, os, json
import numpy as np
import torch
from gpu_weird_soup import run_programs, DEV

OPS = "<>{}-+.,[]"
d = sys.argv[1] if len(sys.argv) > 1 else "runs_pod/weird_soup_char"
pop = np.load(os.path.join(d, "population.npy"))
js = json.load(open(os.path.join(d, "soup.json")))
N, L = pop.shape
print(f"population {pop.shape}  (final zlib trend: {[round(h[1],3) for h in js['hist'][-6:]]})\n")


def decode(row):
    return "".join(OPS[b & 15] if (b & 15) < 10 else "." for b in row)


# top genotypes (the quasispecies)
uniq, cnt = np.unique(pop, axis=0, return_counts=True)
order = np.argsort(-cnt)
print("=== most common genotypes (count / 8192) ===")
for i in order[:6]:
    print(f"  x{int(cnt[i]):4d}  {decode(uniq[i])}")

dom = uniq[order[0]]
print(f"\n=== dominant genotype (x{int(cnt[order[0]])}) decoded ===")
print(f"  code: {decode(dom)}")
print(f"  has copy-op ('.'/',') AND a loop ('[')? {(6 in (dom & 15) or 7 in (dom & 15)) and 8 in (dom & 15)}")

# REPLICATION TEST: concat dominant | random partner, run, measure how much of the partner half became a copy of dominant
print("\n=== replication test: dominant | RANDOM partner -> does it overwrite the partner with itself? ===")
g = torch.Generator(device=DEV).manual_seed(12345)
trials = 256
A = torch.tensor(np.tile(dom, (trials, 1)), device=DEV, dtype=torch.uint8)
B = torch.randint(0, 256, (trials, L), generator=g, device=DEV, dtype=torch.uint8)
before = (B.cpu().numpy() == dom[None, :]).mean()                 # baseline match of random partner to dominant
tape = torch.cat([A, B], dim=1)
tape = run_programs(tape, 2 * (2 * L))                            # give it enough steps to copy
Bafter = tape[:, L:].cpu().numpy()
after = (Bafter == dom[None, :]).mean()                          # match of partner-half to dominant AFTER running
# also: did A-half stay intact (a replicator preserves itself)?
Akept = (tape[:, :L].cpu().numpy() == dom[None, :]).mean()
print(f"  partner-half bytes equal to dominant:  before={before:.3f}  AFTER={after:.3f}   (rise => it copied itself in)")
print(f"  self-half preserved (A still == dominant): {Akept:.3f}")
print(f"\n  VERDICT: {'SELF-REPLICATOR — copies itself into a naive partner (life from noise)' if after > before + 0.25 else 'weak/partial — overlap did not clearly rise (maybe a quasispecies fragment, not a clean copier)'}")
