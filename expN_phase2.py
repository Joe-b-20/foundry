"""Phase 2 of the matmul scaling push (GF(2) first since it's faster + more novel):
(1) GF(2) method validation on 2x2 and 3x3 — does the soft-XOR relaxation find exact mod-2 decompositions
    at all (before attempting the 4x4=47 dream)?
(2) 4x4 REALS wall probe — does the flat CP method find anything sub-naive at 4x4 (T 4096 entries)?"""
import time
import expN_matmul as X
import expN_gf2 as G

DEV = "cpu"

print("########## GF(2) validation: 2x2 (expect rank 7) ##########")
T22 = G.matmul_tensor(2, 2, 2)
t0 = time.time()
for R in range(8, 5, -1):
    bf, exact = G.search_rank(T22, R, 2, 2, 2, 256, 8000, 0.5, 0.05, DEV)
    if exact:
        okm, tot = G.verify_gf2(*exact, 2, 2, 2)
        print(f"  GF2 2x2 R={R}: EXACT mod-2 (verified {okm}/{tot})   [{time.time()-t0:.0f}s]")
    else:
        print(f"  GF2 2x2 R={R}: soft residual {bf:.3e}  (not achieved)   [{time.time()-t0:.0f}s]")

print("\n########## GF(2): 3x3 (best-known mod-2 rank ~23) ##########")
T33 = G.matmul_tensor(3, 3, 3)
t0 = time.time()
for R in range(24, 20, -1):
    bf, exact = G.search_rank(T33, R, 3, 3, 3, 512, 9000, 0.5, 0.05, DEV)
    if exact:
        okm, tot = G.verify_gf2(*exact, 3, 3, 3)
        print(f"  GF2 3x3 R={R}: EXACT mod-2 (verified {okm}/{tot})   [{time.time()-t0:.0f}s]")
    else:
        print(f"  GF2 3x3 R={R}: soft residual {bf:.3e}  (not achieved)   [{time.time()-t0:.0f}s]")

print("\n########## 4x4 REALS wall probe (naive=64; Strassen-recursive=49, but flat CP can't use block structure) ##########")
T44 = X.matmul_tensor(4, 4, 4)
t0 = time.time()
for R in range(64, 58, -1):
    bf, exact = X.search_rank(T44, R, 4, 4, 4, 96, 6000, 0.4, 0.03, DEV)
    if exact:
        okm, tot = X.verify_on_matrices(*exact, 4, 4, 4)
        print(f"  4x4 reals R={R}: EXACT (verified {okm}/{tot})   [{time.time()-t0:.0f}s]")
    else:
        print(f"  4x4 reals R={R}: residual {bf:.2e}  (not achieved)   [{time.time()-t0:.0f}s]")
