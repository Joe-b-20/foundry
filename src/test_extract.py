"""Quick sanity: does extract_loop/interpret/verify_body recover the loop from a KNOWN-GOOD reference
program for mul AND div? (Isolates the extraction machinery from training.)"""
import random
from expG_controller import VM, run_reference, INSTRS, make_problem
import expJ_selfdiscover as J

base = 10
rng = random.Random(0)
for op in ("mul", "div"):
    # build a clean reference rollout at width 3
    A, B, D = make_problem(op, 3, base, rng)
    if op == "div":
        D = max(2, D)
        A = max(A, 100)
    vm = VM(op, A, B, D, base)
    trace = run_reference(vm)
    act = [iid for (_, iid) in trace]
    print(f"\n{op}: reference rollout ({A}{'*'+str(B) if op=='mul' else '/'+str(D)}) "
          f"len={len(act)}: {' '.join(INSTRS[i] for i in act)}")
    body = J.extract_loop(op, act, base, rng)
    if body is None:
        print(f"  extract_loop -> None (FAILED to recover a loop)")
    else:
        bs = " ".join(INSTRS[i] + ("*" if lp else "") for (i, lp) in body)
        print(f"  extract_loop -> body [{bs}]")
        # check the recovered body interprets correctly on fresh problems to w12
        for w in (1, 2, 4, 8, 12):
            ok = 0
            for _ in range(200):
                a, b, d = J._gw(op, w, base, rng)
                _, _, got, halted = J.interpret(op, body, a, b, d, base, cap=(base + 6) * w + 20)
                exp = a * b if op == "mul" else a // d
                ok += (halted and got == exp)
            print(f"    interpret w{w}: {ok/200:.3f}")
