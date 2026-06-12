import importlib
for m in ("mpmath", "sympy", "numpy"):
    try:
        mod = importlib.import_module(m)
        print(m, getattr(mod, "__version__", "?"))
    except Exception as e:
        print(m, "MISSING", e)
try:
    import mpmath
    mpmath.mp.dps = 60
    # quick pslq sanity: find integer relation for [1, sqrt2, sqrt2^2] -> x0*1 + x2*2 = ... (2 - (sqrt2)^2 = 0)
    v = [mpmath.mpf(1), mpmath.sqrt(2), mpmath.sqrt(2) ** 2]
    print("pslq sanity:", mpmath.pslq(v, maxcoeff=10 ** 6))
except Exception as e:
    print("pslq error:", e)
