"""Apify LinkedIn jobs source.

LinkedIn doesn't expose a public jobs API and direct scraping is blocked
from the Routine datacenter IP. The Apify actor `valig/linkedin-jobs-scraper`
runs on Apify's residential infrastructure and returns structured job
listings via the run-sync endpoint.

Actor input schema (verified 2026-05-11 via JSON view in Apify console):
    title:           string (one role per call, e.g. "Director Product AI")
    location:        string (e.g. "India", "Bengaluru")
    datePosted:      string code — "r604800" = past 7 days,
                     "r2592000" = past 30 days
    experienceLevel: array of code strings —
                     "1"=internship, "2"=entry, "3"=associate/mid-senior,
                     "4"=director, "5"=executive/VP, "6"=senior-executive
    contractType:    array — "F"=full-time, "P"=part-time, "C"=contract,
                     "T"=temporary
    remote:          array — "1"=on-site, "2"=remote, "3"=hybrid
    limit:           integer max results per call
    companyName:     array of company strings (optional)
    companyId:       array of LinkedIn company IDs (optional)

Output schema matches `agent/sources/greenhouse.py:Job` so survivors merge
cleanly into `outputs/{date}/_raw_candidates.jsonl` with
`source="linkedin_apify"`.

Pricing: $0.40 per 1,000 results (pay-per-result, not compute-time). At
5 queries × 50 results = 250 results/day = ~$0.10/day = $3/month, well
within Apify's $5/month free credit.

Usage (programmatic):
    from agent.sources.apify_linkedin import run_all_queries
    jobs = run_all_queries()
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

try:
    import requests
except ImportError:
    requests = None  # type: ignore[assignment]


CONFIG_PATH = Path(__file__).resolve().parent.parent / "secrets" / "apify_config.json"
TRAJECTORY_PATH = Path("outputs") / "trajectory.jsonl"
PLACEHOLDER_TOKEN = "REPLACE_WITH_ACTUAL_TOKEN"
TOKEN_ENV_VAR = "APIFY_TOKEN"
APIFY_RUN_URL = (
    "https://api.apify.com/v2/acts/valig~linkedin-jobs-scraper/"
    "run-sync-get-dataset-items"
)

# Date posted codes per the actor's input schema (verified 2026-05-11).
# These are LinkedIn's own filter values, passed verbatim as the actor's
# `datePosted` field.
DATE_PAST_WEEK = "r604800"       # 7 days in seconds
DATE_PAST_MONTH = "r2592000"     # 30 days in seconds

# Senior-band filter: Mid-Senior (3) + Director (4) + Executive/VP (5).
# Excludes junior levels (1, 2) and very-senior CEO/board (6).
EXPERIENCE_SENIOR_BAND = ["3", "4", "5"]

# 24 strategic (title, location) tuples per CLAUDE.md Step 1d spec.
# Covers Director/Principal/GPM/Head/VP titles across India + 3 metros,
# plus Senior/Staff PM at premium-company seniority bands (often
# Principal-equivalent compensation at Google L6 / Databricks Staff /
# Stripe Sr — surfaced because LinkedIn doesn't filter by band).
# Cost: 24 queries × ~30 results × $0.40/1k ≈ $0.30/run = ~$9/month at
# daily cadence. Within Apify $5/month free credit only on alternating
# days — accept a small Apify bill (~$4/mo) for full LinkedIn coverage.
LINKEDIN_QUERIES: list[tuple[str, str]] = [
    # Director / Principal / Group / Head / VP — across India + metros
    ("Director Product Manager", "India"),
    ("Director Product Manager", "Bengaluru"),
    ("Director Product Manager", "Mumbai"),
    ("Director Product Manager", "Hyderabad"),
    ("Principal Product Manager", "India"),
    ("Principal Product Manager", "Bengaluru"),
    ("Principal Product Manager", "Mumbai"),
    ("Group Product Manager", "India"),
    ("Group Product Manager", "Bengaluru"),
    ("Head of Product", "India"),
    ("Head of Product", "Bengaluru"),
    ("VP Product Management", "India"),
    # Senior / Staff PM — premium-company titles (Google L6, Databricks Staff,
    # Stripe Sr, Notion Staff) that the scorer + watchlist filter for fit.
    ("Senior Product Manager", "India"),
    ("Senior Product Manager", "Bengaluru"),
    ("Sr Product Manager", "India"),
    ("Staff Product Manager", "India"),
    ("Staff Product Manager", "Bengaluru"),
    # AI / GenAI / Platform / Agentic focus areas
    ("Director Product AI", "India"),
    ("Principal Product Manager AI", "India"),
    ("Principal Product Manager GenAI", "Bengaluru"),
    ("Director Product Platform", "India"),
    ("Principal Product Manager LLM", "India"),
    ("Director PM Agentic AI", "India"),
    ("Principal PM Developer Productivity", "Bengaluru"),
]


@dataclass
class Job:
    """Mirrors agent/sources/greenhouse.py:Job for downstream compatibility."""
    source: str
    company: str
    title: str
    location: str
    url: str
    posted_at: str | None
    raw_id: str | None
    department: str | None = None
    description_excerpt: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _log_trajectory(record: dict) -> None:
    """Append a JSON line to outputs/trajectory.jsonl. Best-effort, never raises."""
    try:
        TRAJECTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        with TRAJECTORY_PATH.open("a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        # Trajectory logging must never break the run.
        pass


def load_config(path: Path = CONFIG_PATH) -> dict | None:
    """Return resolved config (token + actor + budget + memory), or None."""
    if not path.exists():
        _log_trajectory({
            "step": "1g",
            "source": "linkedin_apify",
            "warning": f"apify_config.json missing at {path}",
            "decision": "skip_apify",
        })
        return None
    try:
        config = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        _log_trajectory({
            "step": "1g",
            "source": "linkedin_apify",
            "warning": f"apify_config.json malformed: {e}",
            "decision": "skip_apify",
        })
        return None

    env_token = (os.environ.get(TOKEN_ENV_VAR) or "").strip()
    file_token = (config.get("apify_token") or "").strip()

    if env_token:
        token = env_token
    elif file_token and file_token != PLACEHOLDER_TOKEN:
        token = file_token
    else:
        _log_trajectory({
            "step": "1g",
            "source": "linkedin_apify",
            "warning": (
                f"no real token: {TOKEN_ENV_VAR} env var unset and "
                f"apify_config.json carries placeholder — set the GitHub "
                f"repo secret APIFY_TOKEN"
            ),
            "decision": "skip_apify",
        })
        return None

    resolved = dict(config)
    resolved["apify_token"] = token
    return resolved


def run_query(title: str, location: str, max_rows: int = 50,
              config: dict | None = None) -> list[dict]:
    """Fire a single Apify run-sync call and return raw dataset items.

    Sends the actor's verified input schema (2026-05-11):
        {title, location, limit, datePosted, experienceLevel}

    Returns an empty list on any failure. Per-query failures are non-fatal
    — caller continues with the next query.
    """
    if config is None:
        config = load_config()
    if config is None:
        return []
    if requests is None:
        _log_trajectory({
            "step": "1g",
            "source": "linkedin_apify",
            "warning": "requests not installed",
            "decision": "skip_apify",
        })
        return []

    token = config["apify_token"]
    memory_mb = config.get("memory_mb", 128)
    params = {"token": token, "memory": memory_mb}

    # Verified actor input schema (Apify console JSON view, 2026-05-11):
    #   title (str), location (str), datePosted (str code), limit (int),
    #   experienceLevel (array of code strings).
    # `rows` is NOT a valid key for this actor — must be `limit`.
    # `datePosted: "Past Week"` is NOT valid — must be `r604800` (seconds).
    payload = {
        "title": title,
        "location": location,
        "limit": max_rows,
        "datePosted": DATE_PAST_WEEK,
        "experienceLevel": EXPERIENCE_SENIOR_BAND,
    }

    try:
        resp = requests.post(APIFY_RUN_URL, params=params, json=payload, timeout=180)
    except Exception as e:
        _log_trajectory({
            "step": "1g",
            "source": "linkedin_apify",
            "query": f"{title} | {location}",
            "warning": f"network error: {type(e).__name__}: {e}",
            "decision": "skip_query_continue",
        })
        return []

    if resp.status_code >= 400:
        _log_trajectory({
            "step": "1g",
            "source": "linkedin_apify",
            "query": f"{title} | {location}",
            "warning": f"http {resp.status_code}: {resp.text[:200]}",
            "decision": "skip_query_continue",
        })
        return []

    try:
        items = resp.json()
    except ValueError as e:
        _log_trajectory({
            "step": "1g",
            "source": "linkedin_apify",
            "query": f"{title} | {location}",
            "warning": f"json decode error: {e}",
            "decision": "skip_query_continue",
        })
        return []

    if not isinstance(items, list):
        return []
    return items


def parse_response(raw_items: list[dict], title: str, location: str) -> list[Job]:
    """Convert Apify dataset items into Job dataclass instances."""
    jobs: list[Job] = []
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        company_name = (
            item.get("companyName")
            or item.get("company")
            or ""
        )
        job_title = item.get("title") or item.get("jobTitle") or ""
        job_loc = item.get("location") or location
        url = (
            item.get("link")
            or item.get("jobUrl")
            or item.get("url")
            or ""
        )
        posted = item.get("postedAt") or item.get("postedDate") or item.get("postedTimeAgo")
        raw_id = item.get("id") or item.get("jobId")
        description = item.get("descriptionText") or item.get("description") or ""
        if isinstance(description, str) and len(description) > 500:
            description = description[:500]

        jobs.append(Job(
            source="linkedin_apify",
            company=str(company_name).strip(),
            title=str(job_title).strip(),
            location=str(job_loc).strip(),
            url=str(url).strip(),
            posted_at=str(posted) if posted is not None else None,
            raw_id=str(raw_id) if raw_id is not None else None,
            department=None,
            description_excerpt=description or None,
        ))
    return jobs


def run_urls(urls: list[str], config: dict | None = None) -> list[dict]:
    """URL-targeted mode — DISABLED.

    valig/linkedin-jobs-scraper does NOT support per-URL targeting. Passing
    `linkedinUrls` causes the actor to ignore the key and run a default
    broad search, consuming budget on irrelevant roles. Verified on
    2026-05-11 run #3: 118 URLs in, 500 random items out, all dropped by
    title filter.

    Kept as a no-op stub for backwards-compat with callers in run_daily.py.
    """
    return []


def run_all_queries(config: dict | None = None,
                    queries: list[tuple[str, str]] | None = None) -> list[dict]:
    """Iterate LINKEDIN_QUERIES and aggregate Job dicts."""
    if config is None:
        config = load_config()
    if config is None:
        return []

    budget = int(config.get("max_jobs_per_run", 250))
    queries = queries if queries is not None else LINKEDIN_QUERIES
    aggregated: list[dict] = []

    for title, location in queries:
        if len(aggregated) >= budget:
            _log_trajectory({
                "step": "1g",
                "source": "linkedin_apify",
                "warning": f"budget cap {budget} reached; stopping",
                "decision": "stop_remaining_queries",
            })
            break
        per_query_room = budget - len(aggregated)
        max_rows = min(50, per_query_room)
        raw_items = run_query(title, location, max_rows=max_rows, config=config)
        jobs = parse_response(raw_items, title, location)
        if len(aggregated) + len(jobs) > budget:
            jobs = jobs[: budget - len(aggregated)]
        aggregated.extend(j.to_dict() for j in jobs)
        _log_trajectory({
            "step": "1g-query",
            "query": f"{title} | {location}",
            "raw_items": len(raw_items),
            "parsed_jobs": len(jobs),
            "running_total": len(aggregated),
        })

    return aggregated


def main() -> int:
    """CLI entry point — useful for manual smoke-testing."""
    jobs = run_all_queries()
    json.dump(jobs, sys.stdout, indent=2, ensure_ascii=False)
    sys.stdout.write("\n")
    print(f"# {len(jobs)} jobs aggregated from LinkedIn via Apify",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
