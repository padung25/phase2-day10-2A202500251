"""Helper script: show circuit breaker transition log for flaky_50 scenario."""
from __future__ import annotations

import random
import sys

sys.path.insert(0, "src")

from reliability_lab.chaos import build_gateway, load_queries
from reliability_lab.config import load_config

cfg = load_config("configs/default.yaml")
queries = load_queries()

gw = build_gateway(cfg, {"primary": 0.5}, False)  # no cache, 50% fail
for _ in range(100):
    gw.complete(random.choice(queries))

for name, breaker in gw.breakers.items():
    if breaker.transition_log:
        print(f"--- {name} transitions ---")
        for t in breaker.transition_log[:10]:
            print(f"  {t['from']:10s} -> {t['to']:10s}  reason={t['reason']}")
