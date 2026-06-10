"""Rectangular ladder: run the batched CP+lattice matmul-rank discovery across several <m,k,p> shapes and
report, per shape, the MINIMUM rank with an EXACT verified integer decomposition = the scaling boundary.
Reuses expN_matmul. Known references (for honest comparison): <2,2,2>=7 (Strassen), <2,2,3>=11, <2,2,4>=14,
<2,3,3>=15, <3,3,3>=23 (Laderman) -- commonly cited best-known ranks."""
import time
import expN_matmul as X

DEVICE = "cpu"
# (m,k,p, Rmax, Rmin, known_best)  -- rectangular ladder (squares 2x2,3x3 characterized separately).
# known-best refs: <2,2,n> = 3n+ceil(n/2)  (=>11,14); <2,3,3>=15 (commonly cited best-known).
CASES = [
    (2, 2, 3, 12, 10, 11),
    (2, 2, 4, 16, 12, 14),
    (2, 3, 3, 18, 13, 15),
]
RESTARTS, STEPS, LAM, LR = 1024, 8000, 0.4, 0.03

for (m, k, p, Rmax, Rmin, known) in CASES:
    T = X.matmul_tensor(m, k, p)
    naive = m * k * p
    print(f"\n=== <{m},{k},{p}>  naive={naive}  best-known≈{known}  T{tuple(T.shape)} ===")
    t0 = time.time()
    min_exact = None
    for R in range(Rmax, Rmin - 1, -1):
        bf, exact = X.search_rank(T, R, m, k, p, RESTARTS, STEPS, LAM, LR, DEVICE)
        if exact:
            okm, tot = X.verify_on_matrices(*exact, m, k, p)
            coeffs = sorted(set(exact[0].unique().tolist() + exact[1].unique().tolist() + exact[2].unique().tolist()))
            print(f"  R={R:3d}: EXACT  (verified {okm}/{tot}; coeffs {coeffs})   [{time.time()-t0:.0f}s]")
            min_exact = R
        else:
            print(f"  R={R:3d}: residual {bf:.2e}  (not achieved)   [{time.time()-t0:.0f}s]")
    if min_exact is not None:
        verdict = "= best-known" if min_exact == known else ("BEATS best-known!" if min_exact < known else f"above best-known ({known})")
        print(f"  -> <{m},{k},{p}> boundary: R={min_exact} (naive {naive}, best-known {known}) [{verdict}]")
    else:
        print(f"  -> <{m},{k},{p}>: no sub-naive exact decomposition found")
