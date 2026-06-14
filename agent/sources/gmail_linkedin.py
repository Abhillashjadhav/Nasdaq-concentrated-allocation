"""Gmail job-alert source — LinkedIn + Naukri (Hirist not present in user's inbox).

LinkedIn doesn't expose a public jobs API and direct scraping is blocked from
the Routine datacenter IP, but the user's Gmail receives daily LinkedIn alert
digests at `jobalerts-noreply@linkedin.com` (and occasionally
`jobs-listings@linkedin.com`). Naukri alerts arrive at
`naukrialerts@naukri.com`. Both are quality-filtered (the user configured the
alerts in-product), so this is a clean candidate signal — same priority as a
direct Indeed query.

Architecture: Gmail MCP is an agent-side tool, not callable from Python. The
agent fetches threads via `Gmail.search_threads` / `get_thread` and dumps the
raw response JSON to `outputs/{date}/_gmail_threads/{linkedin,naukri}.json`
before invoking `fetch_all.py`. This module is parser-only — it reads those
JSON files and emits Job dicts matching `agent/sources/greenhouse.py:Job`.

Output sources (per CLAUDE.md Step 1e):
  - `gmail_linkedin` — alerts from LinkedIn
  - `gmail_naukri`   — alerts from Naukri

Usage (programmatic):
    from agent.sources.gmail_linkedin import fetch_jobs
    jobs = fetch_jobs(date="2026-05-03", root=".")
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path


LINKEDIN_SENDERS = (
    "jobalerts-noreply@linkedin.com",
    "jobs-listings@linkedin.com",
)
NAUKRI_SENDERS = (
    "naukrialerts@naukri.com",
)

# Recommended Gmail MCP queries the agent should use in Step 1e. Both filter
# variants are listed: the label-based one (preferred, set up via Gmail
# filter rules) and the raw `from:` fallback for runs before the user has
# applied filters yet.
LINKEDIN_QUERY_LABEL = "label:JobAlerts/LinkedIn newer_than:2d"
LINKEDIN_QUERY_FROM = (
    "from:jobalerts-noreply@linkedin.com OR "
    "from:jobs-listings@linkedin.com newer_than:2d"
)
NAUKRI_QUERY_LABEL = "label:JobAlerts/Naukri newer_than:2d"
NAUKRI_QUERY_FROM = "from:naukrialerts@naukri.com newer_than:2d"


@dataclass
class Job:
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


# ---- Subject-line parsers --------------------------------------------------

# LinkedIn alerts use two recurring subject formats:
#   1. "{Title} at {Company}"
#      e.g. "Principal Group Product Manager at Microsoft"
#   2. "\"{search_query}\": {Company} - {Title} ( {Location} ) posted on {date}"
#      e.g. '"product management": Smartsheet - Principal Product Manager,
#             Data & AI Platforms ( Hybrid in Bangalore ) posted on 4/30/26'
LINKEDIN_QUOTED_RE = re.compile(
    r"""^\s*[“"']?[^"“”']*[”"']?\s*:\s*  # "search":
        (?P<company>[^\-]+?)\s+-\s+                          # Company -
        (?P<title>.+?)                                       # Title
        (?:\s*\(\s*(?P<location>[^)]+?)\s*\))?               # ( Location )
        (?:\s*posted\s+on\s+(?P<posted_at>[\d/.\-]+))?       # posted on date
        \s*$""",
    re.IGNORECASE | re.VERBOSE,
)
LINKEDIN_AT_RE = re.compile(
    r"^(?P<title>.+?)\s+at\s+(?P<company>[^|]+?)\s*$",
    re.IGNORECASE,
)

# Naukri alert subject:
#   "New job - {Title} | More jobs matching your Custom Job Alert - {alert_name}"
NAUKRI_SUBJECT_RE = re.compile(
    r"^new\s+job\s*[-:]\s*(?P<title>.+?)\s*(?:\||$)",
    re.IGNORECASE,
)

# Naukri snippets pack the title twice in a row before the company:
#   `... matching your job alert {AlertName} {Title} {Title} {Company}
#    (Hybrid|Remote|On-site|Work from Home) - {Location}, ...`
# We use the title from the subject to anchor the rfind, then take the
# words between the last title occurrence and the work-mode keyword as
# the company. NAUKRI_WORKMODE_RE is the right boundary marker.
NAUKRI_WORKMODE_RE = re.compile(
    r"\b(?:Hybrid|Remote|On[-\s]?site|Work\s+from\s+Home)\s*[-–:]\s*"
    r"(?P<location>[A-Za-z][A-Za-z, ]+)",
    re.IGNORECASE,
)


def _job_id_from_url(url: str) -> str | None:
    m = re.search(r"/jobs/view/[^/?#]*?(\d{6,})", url)
    if m:
        return m.group(1)
    m = re.search(r"jobId=(\d+)", url)
    if m:
        return m.group(1)
    return None


def _first_url(text: str) -> str:
    m = re.search(r"https?://[^\s<>\"']+", text or "")
    return m.group(0) if m else ""


def parse_linkedin_subject(subject: str) -> dict | None:
    """Extract title/company/location/posted_at from a LinkedIn alert subject.

    Returns None if neither subject template matches — the caller should
    skip that thread (digest emails with subjects like "View jobs in
    Bengaluru" don't carry a single role and require body parsing, which
    we keep out of scope here).
    """
    if not subject:
        return None
    s = subject.strip()
    # Strip leading "Re:"/"Fwd:" if present
    s = re.sub(r"^(re|fwd|fw):\s*", "", s, flags=re.IGNORECASE)
    m = LINKEDIN_QUOTED_RE.match(s)
    if m:
        return {
            "title": m.group("title").strip(),
            "company": m.group("company").strip(),
            "location": (m.group("location") or "").strip(),
            "posted_at": (m.group("posted_at") or "").strip() or None,
        }
    m = LINKEDIN_AT_RE.match(s)
    if m:
        return {
            "title": m.group("title").strip(),
            "company": m.group("company").strip(),
            "location": "",
            "posted_at": None,
        }
    return None


def parse_naukri_subject_and_snippet(subject: str, snippet: str) -> dict | None:
    """Extract title from Naukri subject + company/location from the snippet."""
    if not subject:
        return None
    title_m = NAUKRI_SUBJECT_RE.match(subject.strip())
    if not title_m:
        return None
    title = title_m.group("title").strip().rstrip(",")

    company = ""
    location = ""
    if snippet:
        # Anchor on the last occurrence of the (clean) subject-title in the
        # snippet — that's the title repetition that immediately precedes
        # the company. Everything between that and the work-mode keyword
        # is the company name.
        idx = snippet.rfind(title)
        after = snippet[idx + len(title):] if idx >= 0 else snippet
        loc_m = NAUKRI_WORKMODE_RE.search(after)
        if loc_m:
            company = after[:loc_m.start()].strip(" -:|")
            location = loc_m.group("location").strip().split(",")[0].strip()

    return {
        "title": title,
        "company": company,
        "location": location,
        "posted_at": None,
    }


# ---- Thread → Job conversion ----------------------------------------------

def parse_linkedin_threads(threads: list[dict]) -> list[Job]:
    """Convert a list of Gmail thread dicts (LinkedIn alerts) to Job entries.

    Expected thread shape mirrors the Gmail MCP `search_threads` /
    `get_thread` response: each thread has a `messages` list, each message
    has `sender`, `subject`, `snippet`, optionally `htmlBody` / `body`.
    Per-message bodies are nice-to-have but not required — the subject line
    of every direct LinkedIn alert (the "{Title} at {Company}" format)
    carries enough signal to score against the JD.
    """
    jobs: list[Job] = []
    for thread in threads or []:
        for msg in thread.get("messages", []):
            sender = (msg.get("sender") or "").lower()
            if not any(s in sender for s in LINKEDIN_SENDERS):
                continue
            parsed = parse_linkedin_subject(msg.get("subject", ""))
            if not parsed:
                continue
            body = msg.get("htmlBody") or msg.get("body") or msg.get("snippet") or ""
            url = _first_url(body) if body else ""
            jobs.append(Job(
                source="gmail_linkedin",
                company=parsed["company"],
                title=parsed["title"],
                location=parsed["location"],
                url=url,
                posted_at=parsed["posted_at"],
                raw_id=_job_id_from_url(url) or msg.get("id"),
                description_excerpt=(msg.get("snippet") or "")[:500] or None,
            ))
    return jobs


def parse_naukri_threads(threads: list[dict]) -> list[Job]:
    jobs: list[Job] = []
    for thread in threads or []:
        for msg in thread.get("messages", []):
            sender = (msg.get("sender") or "").lower()
            if not any(s in sender for s in NAUKRI_SENDERS):
                continue
            parsed = parse_naukri_subject_and_snippet(
                msg.get("subject", ""), msg.get("snippet", ""),
            )
            if not parsed:
                continue
            body = msg.get("htmlBody") or msg.get("body") or msg.get("snippet") or ""
            url = _first_url(body) if body else ""
            jobs.append(Job(
                source="gmail_naukri",
                company=parsed["company"],
                title=parsed["title"],
                location=parsed["location"],
                url=url,
                posted_at=parsed["posted_at"],
                raw_id=msg.get("id"),
                description_excerpt=(msg.get("snippet") or "")[:500] or None,
            ))
    return jobs


# ---- Driver ----------------------------------------------------------------

def fetch_jobs(date: str, root: str | Path = ".") -> list[dict]:
    """Read pre-saved Gmail thread JSON for `date` and return Job dicts.

    Looks for `outputs/{date}/_gmail_threads/linkedin.json` and
    `outputs/{date}/_gmail_threads/naukri.json`. The agent populates these
    files in Step 1e by calling Gmail MCP `search_threads` and dumping the
    raw response. Missing files are non-fatal — return [] for that source
    and continue.

    Each input file should be either:
      - the raw `search_threads` response: `{"threads": [...]}`, OR
      - a bare list of thread dicts.
    """
    root = Path(root)
    base = root / "outputs" / date / "_gmail_threads"
    out: list[dict] = []
    for sender_tag, filename, parser in (
        ("linkedin", "linkedin.json", parse_linkedin_threads),
        ("naukri",   "naukri.json",   parse_naukri_threads),
    ):
        path = base / filename
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        threads = payload.get("threads", payload) if isinstance(payload, dict) else payload
        if not isinstance(threads, list):
            continue
        out.extend(j.to_dict() for j in parser(threads))
    return out


if __name__ == "__main__":  # pragma: no cover
    import argparse
    import sys

    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True)
    ap.add_argument("--root", default=".")
    args = ap.parse_args()
    jobs = fetch_jobs(args.date, args.root)
    json.dump(jobs, sys.stdout, indent=2, ensure_ascii=False)
    sys.stdout.write("\n")
    print(f"# {len(jobs)} jobs from Gmail labels", file=sys.stderr)
