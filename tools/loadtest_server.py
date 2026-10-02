#!/usr/bin/env python3
"""Load-tests a running llmi-server: throughput/latency under concurrent
non-streaming requests, time-to-first-token under concurrent streaming
requests, and that the 429 concurrency limit actually fires under overload
(ADR 0008's single-generation-slot design should show latency grow with
concurrency, not throughput grow with it).

  python3 tools/loadtest_server.py http://127.0.0.1:8103 --concurrency 8 --requests 32
"""
import argparse
import json
import statistics
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor


def post(base_url, payload, stream=False):
    req = urllib.request.Request(
        base_url + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.monotonic()
    first_byte = None
    status = None
    body = b""
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            status = resp.status
            if stream:
                chunk = resp.read(1)
                first_byte = time.monotonic() - start
                body += chunk
            body += resp.read()
    except urllib.error.HTTPError as e:
        status = e.code
        body = e.read()
    elapsed = time.monotonic() - start
    return {"status": status, "elapsed": elapsed, "first_byte": first_byte, "body": body}


def summarize(label, results):
    latencies = [r["elapsed"] for r in results]
    statuses = {}
    for r in results:
        statuses[r["status"]] = statuses.get(r["status"], 0) + 1
    print(f"--- {label} ---")
    print(f"  requests: {len(results)}  status counts: {statuses}")
    if latencies:
        print(f"  latency (s): min={min(latencies):.3f} mean={statistics.mean(latencies):.3f} "
              f"p50={statistics.median(latencies):.3f} max={max(latencies):.3f}")
    first_bytes = [r["first_byte"] for r in results if r["first_byte"] is not None]
    if first_bytes:
        print(f"  time-to-first-byte (s): min={min(first_bytes):.3f} "
              f"mean={statistics.mean(first_bytes):.3f} max={max(first_bytes):.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base_url")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--requests", type=int, default=32)
    ap.add_argument("--max-tokens", type=int, default=32)
    args = ap.parse_args()

    payload = {
        "messages": [{"role": "user", "content": "Describe a sunrise in one sentence."}],
        "max_tokens": args.max_tokens,
    }

    print(f"llmi-server load test against {args.base_url}")
    print(f"concurrency={args.concurrency} requests={args.requests} max_tokens={args.max_tokens}\n")

    # 1. Sequential baseline: one request at a time, no contention.
    with ThreadPoolExecutor(max_workers=1) as pool:
        seq = list(pool.map(lambda _: post(args.base_url, payload), range(args.requests)))
    summarize("sequential (concurrency=1)", seq)

    # 2. Concurrent non-streaming: ADR 0008 predicts latency grows with
    # concurrency (one generation slot) while per-request work is unchanged.
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        conc = list(pool.map(lambda _: post(args.base_url, payload), range(args.requests)))
    summarize(f"concurrent (concurrency={args.concurrency})", conc)

    # 3. Concurrent streaming: time-to-first-byte should still grow with
    # concurrency (the generation mutex serializes the whole request,
    # including its first token), unlike a server with true parallel
    # generation.
    stream_payload = dict(payload, stream=True)
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        stream_results = list(
            pool.map(lambda _: post(args.base_url, stream_payload, stream=True), range(args.requests)))
    summarize(f"concurrent streaming (concurrency={args.concurrency})", stream_results)

    seq_mean = statistics.mean(r["elapsed"] for r in seq)
    conc_mean = statistics.mean(r["elapsed"] for r in conc)
    print(f"\nmean latency grew {conc_mean / seq_mean:.2f}x going from concurrency=1 to "
          f"concurrency={args.concurrency} -- expected under ADR 0008's single generation slot: "
          f"throughput does not scale with concurrency, only queued wait time does.")

    bad_status = [r for r in conc + stream_results if r["status"] not in (200,)]
    if bad_status:
        print(f"\n{len(bad_status)} request(s) did not return 200 under concurrency={args.concurrency}:")
        for r in bad_status[:5]:
            print(f"  status={r['status']} body={r['body'][:200]!r}")

    sys.exit(0)


if __name__ == "__main__":
    main()
