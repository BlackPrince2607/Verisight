"""
VeriSight — Batch Accuracy Test Script
=======================================
Usage:
  1. Put known-REAL images in:  test_data/real/
  2. Put known-FAKE images in:  test_data/fake/
  3. Run: python test_accuracy.py

Supported formats: .jpg .jpeg .png .webp
"""

import os
import time
import json
import requests
from pathlib import Path

# ─── CONFIG ──────────────────────────────────────────────────────────
API_URL       = "http://localhost:8000"
REAL_DIR      = "test_data/real"
FAKE_DIR      = "test_data/fake"
POLL_TIMEOUT  = 60   # max seconds to wait per job
POLL_INTERVAL = 2    # seconds between status checks
RESULTS_FILE  = "accuracy_results.json"
SUPPORTED     = {".jpg", ".jpeg", ".png", ".webp"}

# ─── TERMINAL COLORS ─────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"


# ─── HELPERS ─────────────────────────────────────────────────────────

def upload_file(file_path: Path) -> str | None:
    """Upload a single file, return job_id or None on failure."""
    try:
        with open(file_path, "rb") as f:
            mime = "image/jpeg" if file_path.suffix.lower() in {".jpg", ".jpeg"} else "image/png"
            res  = requests.post(
                f"{API_URL}/upload/",
                files={"file": (file_path.name, f, mime)},
                timeout=30
            )
        if res.status_code == 200:
            return res.json()["job_id"]
        print(f"  {RED}Upload failed ({res.status_code}): {res.text[:80]}{RESET}")
        return None
    except Exception as e:
        print(f"  {RED}Upload error: {e}{RESET}")
        return None


def poll_result(job_id: str) -> dict | None:
    """
    Poll /status/{job_id} until completed/failed or timeout.
    Returns the full status dict on completion, None on failure/timeout.
    """
    elapsed = 0
    while elapsed < POLL_TIMEOUT:
        try:
            res  = requests.get(f"{API_URL}/status/{job_id}", timeout=10)
            data = res.json()

            if data["status"] == "completed":
                return data
            elif data["status"] == "failed":
                return None

        except Exception:
            pass

        time.sleep(POLL_INTERVAL)
        elapsed += POLL_INTERVAL

    return None  # timed out


def evaluate(result: dict, expected: str) -> tuple[str, float, bool]:
    """
    Given a status result and expected label,
    return (got_label, confidence, is_correct).
    """
    raw  = (result.get("result") or result.get("verdict") or "").lower()
    got  = "fake" if "fake" in raw else "real"
    conf = float(result.get("confidence") or 0)
    return got, conf, got == expected


def collect_files(directory: str) -> list[Path]:
    d = Path(directory)
    if not d.exists():
        return []
    return sorted(f for f in d.iterdir() if f.suffix.lower() in SUPPORTED)


def pct(n, total):
    return f"{round(n / total * 100)}%" if total else "—"


def conf_bar(confidence: float, width: int = 20) -> str:
    filled = round(confidence * width)
    bar    = "█" * filled + "░" * (width - filled)
    return bar


# ─── MAIN TEST RUNNER ─────────────────────────────────────────────────

def run_tests():
    print(f"\n{BOLD}{'━'*50}{RESET}")
    print(f"{BOLD}  VeriSight — Accuracy Test Runner{RESET}")
    print(f"{BOLD}{'━'*50}{RESET}")
    print(f"  {DIM}API:      {API_URL}{RESET}")
    print(f"  {DIM}Real dir: {REAL_DIR}{RESET}")
    print(f"  {DIM}Fake dir: {FAKE_DIR}{RESET}\n")

    # ── Check API reachable ──
    try:
        requests.get(f"{API_URL}/health", timeout=5)
        print(f"  {GREEN}✓ API reachable{RESET}\n")
    except Exception:
        print(f"  {RED}✕ Cannot reach API at {API_URL}{RESET}")
        print(f"  {DIM}Make sure both services are running:{RESET}")
        print(f"  {DIM}  cd api_service && python -m uvicorn app.api.main:app --port 8000{RESET}")
        return

    # ── Collect files ──
    real_files = collect_files(REAL_DIR)
    fake_files = collect_files(FAKE_DIR)

    if not real_files and not fake_files:
        print(f"  {RED}✕ No test images found.{RESET}")
        print(f"\n  Create these folders and add images:")
        print(f"  {DIM}  {REAL_DIR}/   ← put real face images here{RESET}")
        print(f"  {DIM}  {FAKE_DIR}/   ← put deepfake images here{RESET}")
        print(f"\n  Good sources:")
        print(f"  {DIM}  Real:  CelebA dataset, news article headshots{RESET}")
        print(f"  {DIM}  Fake:  thispersondoesnotexist.com, FaceForensics++{RESET}\n")
        return

    total_files = len(real_files) + len(fake_files)
    print(f"  Found {BOLD}{len(real_files)}{RESET} real + {BOLD}{len(fake_files)}{RESET} fake = {BOLD}{total_files}{RESET} files\n")

    all_results = []

    # ── Test REAL files ──
    if real_files:
        print(f"{BOLD}  REAL images:{RESET}")
        for f in real_files:
            print(f"    {DIM}{f.name[:40]:<42}{RESET}", end="", flush=True)

            job_id = upload_file(f)
            if not job_id:
                print(f"{RED}UPLOAD FAILED{RESET}")
                all_results.append({"file": f.name, "expected": "real", "got": "error", "correct": False, "confidence": None})
                continue

            result = poll_result(job_id)
            if not result:
                print(f"{YELLOW}TIMEOUT / FAILED{RESET}")
                all_results.append({"file": f.name, "expected": "real", "got": "timeout", "correct": False, "confidence": None})
                continue

            got, conf, correct = evaluate(result, "real")
            icon  = f"{GREEN}✓{RESET}" if correct else f"{RED}✗{RESET}"
            color = GREEN if correct else RED
            bar   = conf_bar(conf)

            print(f"{icon} {color}{got.upper():<5}{RESET}  {bar} {round(conf*100):>3}%")
            all_results.append({"file": f.name, "expected": "real", "got": got, "correct": correct, "confidence": conf})

    # ── Test FAKE files ──
    if fake_files:
        print(f"\n{BOLD}  FAKE images:{RESET}")
        for f in fake_files:
            print(f"    {DIM}{f.name[:40]:<42}{RESET}", end="", flush=True)

            job_id = upload_file(f)
            if not job_id:
                print(f"{RED}UPLOAD FAILED{RESET}")
                all_results.append({"file": f.name, "expected": "fake", "got": "error", "correct": False, "confidence": None})
                continue

            result = poll_result(job_id)
            if not result:
                print(f"{YELLOW}TIMEOUT / FAILED{RESET}")
                all_results.append({"file": f.name, "expected": "fake", "got": "timeout", "correct": False, "confidence": None})
                continue

            got, conf, correct = evaluate(result, "fake")
            icon  = f"{GREEN}✓{RESET}" if correct else f"{RED}✗{RESET}"
            color = GREEN if correct else RED
            bar   = conf_bar(conf)

            print(f"{icon} {color}{got.upper():<5}{RESET}  {bar} {round(conf*100):>3}%")
            all_results.append({"file": f.name, "expected": "fake", "got": got, "correct": correct, "confidence": conf})

    # ── Compute Results ──
    completed = [r for r in all_results if r["got"] not in {"error", "timeout"}]
    correct   = [r for r in completed if r["correct"]]
    errors    = [r for r in all_results if r["got"] in {"error", "timeout"}]

    real_results = [r for r in completed if r["expected"] == "real"]
    fake_results = [r for r in completed if r["expected"] == "fake"]

    real_correct = [r for r in real_results if r["correct"]]
    fake_correct = [r for r in fake_results if r["correct"]]

    conf_values  = [r["confidence"] for r in completed if r["confidence"]]
    avg_conf     = round(sum(conf_values) / len(conf_values) * 100) if conf_values else 0

    overall_acc  = round(len(correct) / len(completed) * 100) if completed else 0
    real_acc     = round(len(real_correct) / len(real_results) * 100) if real_results else 0
    fake_acc     = round(len(fake_correct) / len(fake_results) * 100) if fake_results else 0

    # false positives = real images called fake
    false_pos = [r for r in real_results if not r["correct"]]
    # false negatives = fake images called real
    false_neg = [r for r in fake_results if not r["correct"]]

    # ── Print Report ──
    print(f"\n{BOLD}{'━'*50}{RESET}")
    print(f"{BOLD}  RESULTS{RESET}")
    print(f"{BOLD}{'━'*50}{RESET}\n")

    acc_color = GREEN if overall_acc >= 80 else YELLOW if overall_acc >= 60 else RED
    print(f"  Overall accuracy   {acc_color}{BOLD}{overall_acc}%{RESET}  ({len(correct)}/{len(completed)} correct)")
    print(f"  Real accuracy      {GREEN if real_acc >= 80 else RED}{real_acc}%{RESET}  ({len(real_correct)}/{len(real_results)} correct)")
    print(f"  Fake accuracy      {GREEN if fake_acc >= 80 else RED}{fake_acc}%{RESET}  ({len(fake_correct)}/{len(fake_results)} correct)")
    print(f"  Avg confidence     {avg_conf}%")
    print(f"  False positives    {len(false_pos)}  {DIM}(real images called fake){RESET}")
    print(f"  False negatives    {len(false_neg)}  {DIM}(fake images called real){RESET}")
    if errors:
        print(f"  Errors/timeouts    {YELLOW}{len(errors)}{RESET}")

    # ── Incorrect files ──
    incorrect = [r for r in completed if not r["correct"]]
    if incorrect:
        print(f"\n{BOLD}  Incorrect predictions:{RESET}")
        for r in incorrect:
            conf_str = f"{round(r['confidence']*100)}%" if r["confidence"] else "—"
            print(f"    {RED}✗{RESET} {r['file'][:45]}")
            print(f"      {DIM}expected={r['expected'].upper()}  got={r['got'].upper()}  confidence={conf_str}{RESET}")

    # ── Save JSON ──
    report = {
        "timestamp":       time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_files":     total_files,
        "completed":       len(completed),
        "overall_accuracy": overall_acc,
        "real_accuracy":   real_acc,
        "fake_accuracy":   fake_acc,
        "avg_confidence":  avg_conf,
        "false_positives": len(false_pos),
        "false_negatives": len(false_neg),
        "errors":          len(errors),
        "results":         all_results,
    }

    with open(RESULTS_FILE, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n  {DIM}Full results saved to: {RESULTS_FILE}{RESET}")
    print(f"{BOLD}{'━'*50}{RESET}\n")


if __name__ == "__main__":
    run_tests()