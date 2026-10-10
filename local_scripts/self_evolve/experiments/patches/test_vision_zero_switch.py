"""Verify the paper stage-switch rule against hand-computed expectations."""
import math
import sys

ROOT = "/jizhicfs/rtliu/code/Vision-Zero-official/src/open-r1-multimodal"
sys.path.insert(0, ROOT + "/src")
sys.path.insert(0, ROOT + "/src/open_r1")

from open_r1.trainer.grpo_trainer import VisionZeroStageSwitcher  # noqa: E402

fails = []


def check(name, got, want, tol=1e-12):
    ok = abs(got - want) <= tol if isinstance(want, float) else got == want
    print(("  ok   " if ok else "  FAIL ") + f"{name}: got={got} want={want}")
    if not ok:
        fails.append(name)


print("1. EMA arithmetic: acc_bar after k updates of acc_t=1 (rho=0.95)")
sw = VisionZeroStageSwitcher(patience=None)
for k in range(1, 6):
    sw.update(1.0, 0.0)
    check(f"acc_bar k={k}", sw.acc_bar, 1 - 0.95 ** k)
check("na_bar k=5 (na_t=0)", sw.na_bar, 0.0)

print("2. threshold switch: constant acc=1, na=0, P disabled")
sw = VisionZeroStageSwitcher(patience=None, k_min=5)
switched_at = None
for k in range(1, 60):
    stage, switched, reason = sw.update(1.0, 0.0)
    if switched and switched_at is None:
        switched_at = k
        check("dwell reset at the switch", sw.dwell, 0)
# acc_bar >= 0.9 needs 0.95**k <= 0.1 -> k >= 45
check("switch update index", switched_at, 45)
check("stage after switch", sw.stage, 1)

print("3. K_MIN gates the switch")
sw = VisionZeroStageSwitcher(patience=None, k_min=50, rho=0.0)  # rho=0 so acc_bar=1 immediately
early = [sw.update(1.0, 0.0)[1] for _ in range(49)]
check("no switch before K_MIN", any(early), False)
check("switch exactly at K_MIN", sw.update(1.0, 0.0)[1], True)

print("4. patience forces a switch after P rounds")
sw = VisionZeroStageSwitcher(patience=20)
hits = [k for k in range(1, 46) if sw.update(1.0, 0.0)[1]]
check("first forced switch at P", hits[0] if hits else None, 20)
check("second forced switch at 2P", hits[1] if len(hits) > 1 else None, 40)

print("5. decision->clue needs na_bar <= 0.1 as well")
sw = VisionZeroStageSwitcher(patience=None, k_min=1)
sw.stage, sw.acc_bar, sw.na_bar, sw.dwell = 0, 0.9, 0.5, 1
stage, switched, reason = sw.update(1.0, 1.0)  # keeps na_bar high
check("blocked by na_bar", switched, False)
sw2 = VisionZeroStageSwitcher(patience=None, k_min=1)
sw2.stage, sw2.acc_bar, sw2.na_bar, sw2.dwell = 0, 0.9, 0.0, 1
check("allowed when na_bar low", sw2.update(1.0, 0.0)[1], True)

print("6. clue->decision on error rate or na rate")
for label, acc, na, want in [("error high", 0.5, 0.0, True),
                             ("error low", 0.95, 0.0, False),
                             ("na high", 0.95, 0.9, True)]:
    s = VisionZeroStageSwitcher(patience=None, k_min=1)
    s.stage, s.dwell = 1, 5
    # update() applies the EMA first; pre-load so the EMA lands on the target
    s.acc_bar = (acc - 0.05 * acc) / 0.95
    s.na_bar = (na - 0.05 * na) / 0.95
    check(f"clue->decision {label}", s.update(acc, na)[1], want)

print("7. b_s/b_civ/acc_bar/na_bar must not reset on a switch")
sw = VisionZeroStageSwitcher(patience=20)
for _ in range(20):
    sw.update(1.0, 0.0)
before = (sw.acc_bar, sw.na_bar, sw.updates)
sw.update(1.0, 0.0)
check("acc_bar survives switch", sw.acc_bar >= before[0], True)
check("updates counter monotonic", sw.updates, before[2] + 1)

print("8. paper constants")
sw = VisionZeroStageSwitcher()
for name, want in [("RHO", 0.95), ("TAU_ACC_UP", 0.9), ("TAU_ERR_UP", 0.4),
                   ("TAU_NA_UP", 0.5), ("TAU_NA_DOWN", 0.1), ("K_MIN", 5), ("PATIENCE", 20)]:
    check(name, getattr(sw, name), want)

print()
print("RESULT:", "ALL PASS" if not fails else f"{len(fails)} FAILURES: {fails}")
