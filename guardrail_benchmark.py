#!/usr/bin/env python3
import argparse, csv, json, random, time, statistics, collections
import requests
from typing import Dict, Any, List

# ---------------------------------------
# Prompt generators (good / short / noisy)
# ---------------------------------------
DOMAINS = [
    "ride-sharing app", "food delivery platform", "online bookstore",
    "banking portal", "IoT home automation system", "university course registration",
    "video streaming service", "ticketing system", "health records system",
    "chat application",
]

CAPABILITIES = [
    "users sign up and log in",
    "riders request rides and drivers accept",
    "customers place orders and track status",
    "payments processed via external gateway",
    "search and browse items",
    "messages exchanged in real time",
    "stream media over CDN",
    "appointments scheduled and reminders sent",
    "files uploaded and stored in a database",
]

def prompt_good() -> str:
    d = random.choice(DOMAINS)
    c1, c2 = random.sample(CAPABILITIES, 2)
    return (
        f"A {d} where {c1}; {c2}. "
        f"Data is stored in a database and sensitive flows cross a trust boundary "
        f"to an external payment/API. Include actors, processes, data stores, and flows."
    )

def prompt_short() -> str:
    d = random.choice(DOMAINS)
    return f"{d}. make a diagram."

def prompt_noisy() -> str:
    d = random.choice(DOMAINS)
    return (
        f"pls ASAP create a dfd!!! need arrows etc for {d},"
        f" just do it quickly???? thx"
    )

def generate_prompts(n: int) -> List[str]:
    prompts = []
    for _ in range(n):
        r = random.random()
        if r < 0.6:
            prompts.append(prompt_good())
        elif r < 0.8:
            prompts.append(prompt_short())
        else:
            prompts.append(prompt_noisy())
    return prompts

# ---------------------------------------
# HTTP helpers
# ---------------------------------------
def post_json(url: str, body: Dict[str, Any], timeout: float = 15.0) -> (Dict[str, Any], float):
    t0 = time.perf_counter()
    resp = requests.post(url, json=body, timeout=timeout)
    dt = time.perf_counter() - t0
    resp.raise_for_status()
    return resp.json(), dt

# ---------------------------------------
# /refine-prompt benchmark
# ---------------------------------------
def run_refine_benchmark(api_base: str, n: int, label: str) -> List[Dict[str, Any]]:
    url = api_base.rstrip("/") + "/refine-prompt"
    prompts = generate_prompts(n)
    rows = []

    for i, p in enumerate(prompts, 1):
        try:
            data, dt = post_json(url, {"text": p})
            valid = bool(data.get("valid"))
            refined = data.get("refined_prompt", "")

            # collect error/hints summary (first item as “reason” string to keep CSV tidy)
            reason = ""
            if not valid:
                errs = data.get("errors") or []
                if errs and isinstance(errs, list):
                    reason = errs[0].get("message", str(errs[0]))[:160]
                else:
                    hints = data.get("hints") or []
                    if hints:
                        reason = str(hints[0])[:160]

            rows.append({
                "idx": i,
                "label": label,
                "latency_s": round(dt, 4),
                "valid": valid,
                "reason_or_hint": reason,
                "prompt": p,
                "refined": refined,
            })
        except Exception as e:
            rows.append({
                "idx": i,
                "label": label,
                "latency_s": -1,
                "valid": False,
                "reason_or_hint": f"HTTP_ERROR: {e}",
                "prompt": p,
                "refined": "",
            })

    return rows

def summarize(rows: List[Dict[str, Any]], label: str) -> None:
    latencies = [r["latency_s"] for r in rows if r["latency_s"] >= 0]
    valids = [r for r in rows if r["valid"]]
    invalids = [r for r in rows if not r["valid"]]

    print("\n=== Refine Benchmark Summary:", label, "===")
    print("Total:", len(rows))
    print("Valid:", len(valids), f"({len(valids)/len(rows)*100:.1f}%)")
    print("Invalid:", len(invalids), f"({len(invalids)/len(rows)*100:.1f}%)")

    if latencies:
        p50 = statistics.quantiles(latencies, n=100)[49]
        p95 = statistics.quantiles(latencies, n=100)[94]
        print(f"Latency: mean={statistics.mean(latencies):.3f}s  p50={p50:.3f}s  p95={p95:.3f}s")

    reasons = collections.Counter(
        (r["reason_or_hint"] or "UNKNOWN") for r in invalids
    ).most_common(5)
    if reasons:
        print("Top invalid reasons/hints:")
        for msg, cnt in reasons:
            print(f"  {cnt:3d}  {msg}")

def save_csv(rows: List[Dict[str, Any]], label: str) -> str:
    fname = f"refine_results_{label}.csv"
    with open(fname, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("Saved:", fname)
    return fname

# ---------------------------------------
# /validate-json smoke tests
# ---------------------------------------
def run_validate_smoke(api_base: str) -> None:
    url = api_base.rstrip("/") + "/validate-json"

    # Canonical payload that should PASS (or only warn)
    canonical_ok = {
  "version": "1.0.0",
  "metadata": {"title": "Smoke DFD"},
  "nodes": {
    "processes": [
      {"id": "p1", "name": "API", "trust_boundary_id": "tb-internal"}
    ],
    "data_stores": [
      {"id": "ds1", "name": "DB", "trust_boundary_id": "tb-internal"}
    ],
    "external_entities": [
      {"id": "ext1", "name": "Payment GW", "trust_boundary_id": "tb-internet"}
    ]
  },
  "trust_boundaries": [
    {"id": "tb-internet", "name": "Internet"},
    {"id": "tb-internal", "name": "Internal"}
  ],
  "flows": [
    {"id": "f1", "source": "ext1", "target": "p1", "data": "payment request", "protocol": "HTTPS"},
    {"id": "f2", "source": "p1", "target": "ds1", "data": "transaction record write", "protocol": "TCP"}
  ]
}


    # RAW payload designed to MAP -> Canonical and then PASS validation
    raw_like = {
        "components": [
            {"id": "u1", "type": "oval", "data": {"text": "User"}},          # -> external_entity
            {"id": "svc", "type": "rect", "data": {"text": "Service"}},      # -> process
            {"id": "db1", "type": "document", "data": {"text": "Database"}}, # -> data_store
        ],
        "connections": [
            {
                "id": "c1",
                "from": "u1",
                "to": "svc",
                "data": {"startLabel": "request", "endLabel": "", "protocol": "HTTPS"}
            },
            {
                "id": "c2",
                "from": "svc",
                "to": "db1",
                "data": {"startLabel": "write", "endLabel": "record", "protocol": "TCP"}
            }
        ]
    }

    # Malformed on purpose — should FAIL with clear errors
    malformed = {"weird": "stuff"}

    for name, payload in [("canonical_ok", canonical_ok), ("raw_like", raw_like), ("malformed", malformed)]:
        try:
            data, dt = post_json(url, payload)
            print(f"\nValidate {name}: {dt:.3f}s  valid={data.get('valid')}")
            if not data.get("valid"):
                for e in (data.get("errors") or [])[:5]:
                    print(" -", e.get("message", str(e))[:200])
        except Exception as e:
            print(f"\nValidate {name}: HTTP_ERROR {e}")


# ---------------------------------------
# CLI
# ---------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", required=True, help="Base URL, e.g., http://guardrail-alb-....ap-southeast-2.elb.amazonaws.com")
    ap.add_argument("--n", type=int, default=100, help="Number of prompts")
    ap.add_argument("--label", default="run", help="Label for this run (e.g., no-llm, llm-on)")
    ap.add_argument("--skip-validate", action="store_true", help="Skip /validate-json smoke tests")
    args = ap.parse_args()

    rows = run_refine_benchmark(args.api, args.n, args.label)
    summarize(rows, args.label)
    save_csv(rows, args.label)

    if not args.skip_validate:
        run_validate_smoke(args.api)
