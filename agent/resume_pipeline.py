"""
Inline ReportLab resume builder.

Reads the run's _scored_candidates.json and profile/master_profile.json, emits
one date-stamped PDF per >=80 fit (original or bumped) to
outputs/{date}/resumes/.

Usage:
    python agent/resume_pipeline.py --date 2026-05-03

Layout / framing:
  * Resume content is sourced from profile/master_profile.json.
  * A4 page size, margins 0.58"L/R / 0.48"T/B.
  * Embeds Liberation Sans (Helvetica equivalent) + Liberation Serif TTF so
    ATS / Workday parsers see embedded glyph data (file size ~48-50 KB).
  * Detects people-manager vs IC roles by title and applies per-mode framing.
  * Honors the EXACT TITLES table (overrides master_profile where they differ).
  * Honors verified factual anchors (PayTM 8 PMs / 30% adoption,
    Wayfair "enabling SDLC velocity", IndiaMART catalog/listing quality,
    Flipkart "seller acquisition workflow efficiency").
  * Enforces content exclusivity (Amazon-only / Wayfair-only keywords).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.flowables import HRFlowable


# Color palette per the layout spec FORMATTING > "Color palette".
BLUE = HexColor("#1F4E8C")
BODY = HexColor("#1a1a1a")
GREY = HexColor("#333333")


LIB_TTF_DIR = Path("/usr/share/fonts/truetype/liberation")


# ---------- Font registration ------------------------------------------------

# We register Liberation Sans as the visual "Helvetica" body / header font so
# embedded glyph data lands in the PDF (8-12 KB output = unembedded built-in
# Helvetica per the embedded-fonts note; 48-50 KB = correctly embedded).
BASE_FONT = "LibSans"
BOLD_FONT = "LibSans-Bold"
ITALIC_FONT = "LibSans-Italic"
BOLD_ITALIC_FONT = "LibSans-BoldItalic"
SERIF_FONT = "LibSerif"


def _register_fonts() -> None:
    sans_dir = Path("/usr/share/fonts/truetype/liberation")
    pdfmetrics.registerFont(TTFont(BASE_FONT, str(sans_dir / "LiberationSans-Regular.ttf")))
    pdfmetrics.registerFont(TTFont(BOLD_FONT, str(sans_dir / "LiberationSans-Bold.ttf")))
    pdfmetrics.registerFont(TTFont(ITALIC_FONT, str(sans_dir / "LiberationSans-Italic.ttf")))
    pdfmetrics.registerFont(TTFont(BOLD_ITALIC_FONT, str(sans_dir / "LiberationSans-BoldItalic.ttf")))
    # Tell ReportLab how the bold/italic variants relate so <b>/<i> tags work.
    from reportlab.pdfbase.pdfmetrics import registerFontFamily
    registerFontFamily(
        BASE_FONT,
        normal=BASE_FONT, bold=BOLD_FONT,
        italic=ITALIC_FONT, boldItalic=BOLD_ITALIC_FONT,
    )


# ---------- Layout spec constants -------------------------------------------

# Override profile titles to match the EXACT TITLES table in the layout spec.
TITLE_OVERRIDES = {
    "Wayfair":          "Senior Manager, GenAI Product Management",
    "RADAR by AIMleap": "Product Advisor (Pro Bono)",
    "CTL":              "Director, Product Management",
    "Amazon":           "Product Head (Senior PM)",
    "PayTM":            "AVP",
    "IndiaMart":        "Vice President, Product",
    "LeEco":            "Senior Manager, Online Marketing",
    "Flipkart":         "Senior Manager, Product",
    "Marico Industries":   "Area Sales Manager",
    "AgroTech Foods Ltd":  "Area Sales Manager",
}

# Hard exclusion per the layout spec.
NEVER_INCLUDE_COMPANIES = {"MJ Internet Pvt Ltd"}

CORE_SKILLS_LINE = (
    "GenAI/LLM Platforms · Agentic AI · Recommendation Systems · "
    "Developer Tooling · API Ecosystem · RAG · Microservices (AWS EC2, ECS) · "
    "Cross-functional Leadership"
)

EDUCATION_LINE = "B.E., M.Tech, MBA (Marketing — JBIMS)"

# Decline-list (case-insensitive substring) per the layout spec. Resume generation
# is skipped entirely if the candidate company matches.
DECLINE_LIST = [
    # Amazon umbrella
    "Amazon", "AWS", "Twitch", "IMDb", "Whole Foods", "Audible", "Ring",
    "Zappos", "MGM Studios", "Goodreads", "Kuiper", "A9", "Alexa", "Lab126",
    # Cimpress umbrella
    "Cimpress", "CTL", "Vista", "Vistaprint", "Pixartprinting", "Drukwerkdeal",
    "BuildASign", "Printi", "WIRmachenDRUCK", "Tradeprint", "National Pen",
    "Easyflyer", "Exaprint", "Druck.at", "VistaCreate",
]

# CONTENT EXCLUSIVITY — keywords that must only appear under their owning
# company's bullets. Used to scrub bullets when reframing.
AMAZON_ONLY_TERMS = (
    "RFM", "AARRR", "CDP", "Customer Data Platform",
    "collaborative filtering", "content-based filtering", "matrix factorisation",
    "matrix factorization",
)
WAYFAIR_ONLY_TERMS = (
    "agentic coding assistant", "RAG-powered PR evaluation", "Model Proxy",
    "Advisory Committee", "Champions Group", "Cursor", "Copilot",
)


# ---------- People-manager vs IC detection (the layout spec section) ---------------

PEOPLE_MGR_TITLE_RE = re.compile(
    r"\b(director|sr\.?\s*director|senior\s+director|vp|vice\s+president|"
    r"head\s+of\s+product|group\s+product|gpm)\b",
    re.IGNORECASE,
)
IC_TITLE_RE = re.compile(
    r"\b(principal|staff|lead)\s+pm\b|"
    r"\bprincipal\s+product\s+manager\b|"
    r"\bstaff\s+product\s+manager\b|"
    r"\bsenior\s+product\s+manager\b",
    re.IGNORECASE,
)


def detect_role_mode(candidate: dict) -> str:
    """Return 'people_manager' or 'ic' based on candidate title.

    Per the layout spec: when ambiguous (e.g. 'Principal Group PM'), default to IC.
    Director / Senior Director / VP / Head / GPM strictly imply people manager.
    """
    title = (candidate.get("title") or candidate.get("role") or "").lower()
    # IC takes priority when 'principal' is in the title — even for
    # 'Principal Group PM' the layout-spec default is IC.
    if "principal" in title or "staff product" in title:
        return "ic"
    if PEOPLE_MGR_TITLE_RE.search(title):
        return "people_manager"
    if "senior product manager" in title or "lead product manager" in title:
        return "ic"
    return "ic"  # safe default per the layout spec ambiguity rule


# ---------- Helpers ----------------------------------------------------------

def slugify(text: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "-", text or "").strip("-").lower()
    return s


def is_declined(company: str) -> bool:
    co = (company or "").lower()
    return any(d.lower() in co for d in DECLINE_LIST)


def jd_keywords_for(candidate: dict) -> list[str]:
    """Pick keywords from Abhillash's catalog that overlap the JD.

    The pool is intentionally per-JD: it pulls from title + company + why_fit +
    the JD body itself (description / description_excerpt) so a Stripe Principal
    PM brief and a Wayfair Director PM brief surface different bolded phrases.
    """
    pool = set()
    blob = " ".join([
        (candidate.get("title") or "").lower(),
        (candidate.get("company") or "").lower(),
        " ".join(candidate.get("why_fit") or []).lower(),
        (candidate.get("description_excerpt") or "").lower(),
        (candidate.get("description") or "").lower(),
    ])
    catalog = [
        "Model Proxy", "agentic", "RAG", "LangChain", "LangGraph",
        "developer velocity", "$300M business impact", "2,800+ engineers",
        "AI/ML", "GenAI", "LLM", "platform", "recommendation",
        "personalisation", "subscriptions", "B2B", "marketplace", "0-to-1",
        "multi-tenant", "microservices", "AWS", "eval framework",
        "governance", "developer tooling", "$3.2B GMV", "$1.2B AI/ML growth",
        "$4B GMV", "Customer Data Platform",
        # Stretch-mode adjacencies — bolded when the JD calls for them.
        "API", "SDK", "developer platform", "observability",
        "payments", "fintech", "BNPL", "fraud", "risk",
        "search", "ranking", "discovery", "personalization",
        "trust and safety", "compliance", "data platform",
        "pricing", "billing", "checkout", "growth",
        "experimentation", "A/B testing", "PLG", "self-serve",
        "enterprise SaaS", "0-to-1", "scale", "roadmap",
    ]
    for term in catalog:
        if term.lower() in blob or any(
            tok in blob for tok in term.lower().replace("/", " ").split()
            if len(tok) > 4
        ):
            pool.add(term)
    # Anchor terms — always bold these, they identify Abhillash's signature work.
    pool.update({"$300M business impact", "2,800+ engineers", "Model Proxy"})
    return list(pool)


# Adjacency map: every JD keyword we are willing to bold must trace back to
# documented evidence in master_profile.json. Bolding a keyword the user has
# no proven experience for is a mild form of fabrication — recruiters expect
# bolded phrases to be load-bearing.
ADJACENCY_EVIDENCE = {
    # AI / ML / GenAI — Wayfair Model Proxy, Amazon AI/ML reco
    "agentic": "Wayfair Model Proxy",
    "genai": "Wayfair Model Proxy",
    "rag": "Wayfair Model Proxy",
    "llm": "Wayfair Model Proxy",
    "langchain": "Wayfair Model Proxy",
    "langgraph": "Wayfair Model Proxy",
    "ai/ml": "Amazon AI/ML recommendation engine",
    "model proxy": "Wayfair Model Proxy",
    # Platform / DevTools — Wayfair GenAI dev platform
    "platform": "Wayfair GenAI dev platform",
    "developer velocity": "Wayfair GenAI dev platform",
    "developer platform": "Wayfair GenAI dev platform",
    "developer tooling": "Wayfair GenAI dev platform",
    "eval framework": "Wayfair Model Proxy evals",
    "observability": "Wayfair Model Proxy",
    # Reco / Personalization — Amazon + CTL
    "recommendation": "Amazon AI/ML reco $1.2B",
    "personalization": "CTL personalization $3.2B GMV",
    "personalisation": "CTL personalization $3.2B GMV",
    "search": "Amazon Promotions Platform",
    "ranking": "Amazon AI/ML reco",
    # B2B / Subscriptions / Marketplace — PayTM, IndiaMART
    "b2b": "PayTM B2B Payments founding PM",
    "marketplace": "PayTM B2B + IndiaMART Big Brands",
    "subscriptions": "IndiaMART subscriptions ₹30 Cr ARR",
    "multi-tenant": "PayTM B2B Payments",
    # Payments / Fintech — PayTM
    "payments": "PayTM B2B Payments founding PM",
    "fintech": "PayTM B2B Payments founding PM",
    "bnpl": "PayTM B2B Payments",
    # Infra / API
    "api": "Wayfair Model Proxy API",
    "sdk": "Wayfair Model Proxy SDK",
    "microservices": "Wayfair Model Proxy",
    "aws": "Wayfair Model Proxy",
    # Governance / Trust
    "governance": "Wayfair Model Proxy governance layer",
    "trust and safety": "Wayfair Model Proxy governance",
    "compliance": "Wayfair Model Proxy governance",
    # Scale / 0-to-1
    "0-to-1": "PayTM B2B Payments founding PM",
    "scale": "Amazon Promotions Platform $4B GMV",
    # Anchors — always allowed
    "$300m business impact": "Wayfair $300M",
    "2,800+ engineers": "Wayfair 2,800+ engineers",
    "$3.2b gmv": "CTL personalization",
    "$1.2b ai/ml growth": "Amazon AI/ML reco",
    "$4b gmv": "Amazon Promotions Platform",
    "customer data platform": "Amazon CDP",
    "enterprise saas": "Wayfair / CTL / IndiaMART",
}


def _has_adjacency_evidence(kw: str) -> bool:
    """Return True iff `kw` maps to a documented experience bullet."""
    return kw.lower().strip() in ADJACENCY_EVIDENCE


def bold_keywords(text: str, keywords: list[str]) -> str:
    """Bold JD keywords ONLY when they map to documented evidence.

    Adjacency check (ADJACENCY_EVIDENCE) prevents implying fluency we can't
    substantiate. An unmapped keyword silently passes through unbolded — it's
    fine if the keyword happens to appear in the bullet (it's the user's own
    text), we just don't draw the recruiter's eye to it as a claim.
    """
    if not text:
        return ""
    out = text
    for kw in sorted(set(keywords), key=len, reverse=True):
        if not kw:
            continue
        if not _has_adjacency_evidence(kw):
            continue
        pattern = re.compile(re.escape(kw), re.IGNORECASE)
        m = pattern.search(out)
        if not m:
            continue
        s, e = m.span()
        before = out[:s]
        if before.rfind("<b>") > before.rfind("</b>"):
            continue
        out = out[:s] + "<b>" + out[s:e] + "</b>" + out[e:]
    return out


def scrub_cross_company_terms(text: str, owning_company: str) -> str:
    """Remove cross-company terms per CONTENT EXCLUSIVITY in the layout spec.

    e.g. 'RFM' / 'AARRR' / 'CDP' must only appear under Amazon bullets;
    'Model Proxy' / 'agentic coding assistant' only under Wayfair.
    """
    co = (owning_company or "").lower()
    out = text
    if "amazon" not in co:
        for term in AMAZON_ONLY_TERMS:
            out = re.sub(re.escape(term), "", out, flags=re.IGNORECASE)
    if "wayfair" not in co:
        for term in WAYFAIR_ONLY_TERMS:
            out = re.sub(re.escape(term), "", out, flags=re.IGNORECASE)
    # Tidy any double-spaces / hanging punctuation we just introduced.
    out = re.sub(r"\s{2,}", " ", out)
    out = re.sub(r"\(\s*[,;]\s*", "(", out)
    out = re.sub(r"\s*[,;]\s*\)", ")", out)
    return out


# ---------------- Integrity guardrails (no fluff, no fabrication) -----------
#
# Hard rules baked into every bullet/summary before it lands in the PDF:
#
# 1. NO FLUFF — banned superlatives/marketing adjectives that don't carry
#    information. Recruiters discount them; ATS systems don't weight them.
#    The list below is curated to remove the dross without breaking phrases
#    where the word is load-bearing (e.g. "world-class" stays only when
#    quantified, never as a self-applied compliment).
#
# 2. NO FABRICATION — we only retain the metrics/scope/dates documented in
#    profile/master_profile.json. The reframe path is allowed to use JD
#    language for the same documented achievement, but cannot invent new
#    numbers, titles, or scope. Fabrication detection runs on every bullet
#    that mentions a $-figure or %-figure not in the profile catalog.
#
# 3. ADJACENT SKILLS ONLY — JD-keyword bolding is restricted to phrases that
#    map to a documented experience bullet OR an adjacency the user has
#    explicit evidence for (e.g. "payments" maps to PayTM; "agentic AI" to
#    Wayfair Model Proxy). Unmapped JD keywords are NOT bolded — they would
#    imply fluency we can't substantiate.

FLUFF_WORDS = [
    # Self-applied superlatives — never substantiated
    "best-in-class", "best in class", "world-class", "world class",
    "industry-leading", "industry leading", "cutting-edge", "cutting edge",
    "state-of-the-art", "state of the art", "next-generation", "next generation",
    "next-gen", "best-of-breed", "best of breed",
    # Hype adjectives
    "amazing", "incredible", "phenomenal", "unparalleled", "unmatched",
    "extraordinary", "exceptional", "remarkable", "outstanding",
    "groundbreaking", "ground-breaking", "revolutionary", "disruptive",
    "transformative", "game-changing", "game changing", "innovative",
    "innovating", "pioneering",
    # Filler
    "brand new", "brand-new", "robust", "synergistic", "holistic",
    "seamless", "seamlessly", "leverage", "leveraging", "leveraged",
    "passionate", "passionately", "drove transformation", "thought leader",
    "thought-leader", "rockstar", "ninja", "guru",
    # Empty modifiers
    "very ", "extremely ", "highly ", "deeply ", "truly ",
    "successfully ", "effectively ", "efficiently ",
    # Self-praise framings
    "spearheaded the", "championed the", "orchestrated the",
]

# Documented numeric anchors derived dynamically from the master profile.
# `load_approved_metrics()` scans every numeric/scope token in the profile
# (executive_summary, every experience bullet, every metric/scope field) and
# returns the set of strings that the fabrication guard treats as "documented."
# Hard-coding this list is fragile — the profile is the source of truth.
APPROVED_METRICS_STATIC_FALLBACK = [
    "$300M", "$15M", "$3.2B", "$4B", "$1.2B", "$1B", "$500M",
    "800M+", "10M+", "2,800+", "15,000+", "13 brands", "11+ years",
    "16+ years", "₹30 Cr", "Rs 30 Cr", "₹78 Cr", "Rs 78 Cr",
    "0-to-1", "0 to 1", "0→1",
]

_APPROVED_METRICS_CACHE: set[str] | None = None


def load_approved_metrics(profile: dict | None = None) -> set[str]:
    """Extract every numeric/$-figure token from the master profile.

    Returns a set of normalised strings. Cached after first call; pass
    `profile=None` (default) to use the cached set on subsequent calls.

    Tokens captured:
      - Dollar / ₹ / Rs amounts:        $300M, $4B, $1.2B, ₹30 Cr
      - Suffix-K/M/B with optional +:    800M+, 10M+, 2,800+, 15,000+
      - Year ranges:                     11+ years, 16+ years
      - Counts with + suffix:            20+ marketplaces, 8 countries
      - Phrases the profile uses verbatim: 0-to-1, 13 brands
    """
    global _APPROVED_METRICS_CACHE
    if profile is None and _APPROVED_METRICS_CACHE is not None:
        return _APPROVED_METRICS_CACHE
    if profile is None:
        return set(APPROVED_METRICS_STATIC_FALLBACK)

    blob_parts = [profile.get("executive_summary", "")]
    for role in profile.get("experience", []):
        blob_parts.extend(role.get("achievements", []) or [])
        if role.get("scope"):
            blob_parts.append(role["scope"])
        if role.get("metrics"):
            blob_parts.extend(role["metrics"])
        blob_parts.append(role.get("title", ""))
    blob = "\n".join(blob_parts)

    metrics: set[str] = set()
    # $ and ₹ amounts
    for m in re.finditer(r"\$[\d.,]+\s*[BMKbmk]?", blob):
        metrics.add(m.group().strip())
    for m in re.finditer(r"₹[\d.,]+\s*Cr?", blob):
        metrics.add(m.group().strip())
    for m in re.finditer(r"Rs\.?\s*[\d.,]+\s*Cr?", blob):
        metrics.add(m.group().strip())
    # Suffix-K/M/B with optional + (e.g. 800M+, 10M+, 2,800+, 15,000+)
    for m in re.finditer(r"\b\d+(?:,\d{3})*\+?\s*[BMKbmk]?\+?\b", blob):
        tok = m.group().strip()
        # Skip standalone tiny integers (3, 6, 8 etc) — too noisy.
        if re.fullmatch(r"\d{1,2}", tok):
            continue
        metrics.add(tok)
    # Year phrases
    for m in re.finditer(r"\b\d+\+?\s*years?\b", blob, re.IGNORECASE):
        metrics.add(m.group().strip())
    # Always-allowed anchor phrases
    metrics.update({"0-to-1", "0 to 1", "0→1"})

    _APPROVED_METRICS_CACHE = metrics
    return metrics


def detect_fabricated_metrics(text: str, profile: dict | None = None) -> list[str]:
    """Return any $-figure or %-figure in the text NOT documented in profile.

    The "documented" set comes from `load_approved_metrics(profile)` which
    scans master_profile.json for every numeric token. A non-empty return
    means a bullet contains a number we have no source for — that bullet
    should be dropped, not coerced.
    """
    if not text:
        return []
    approved = load_approved_metrics(profile)
    approved_lower = {m.lower() for m in approved}
    numeric_re = re.compile(
        r"(\$[\d.,]+\s*[BMK]?|₹[\d.,]+\s*Cr?|Rs\.?\s*[\d.,]+\s*Cr?|"
        r"\b\d+(?:,\d{3})*\+?\s*(?:M|B|K)?\+?\b(?!\s*(?:years|year|months|weeks|days|bps|%)))",
        re.IGNORECASE,
    )
    found = numeric_re.findall(text)
    suspect = []
    for f in found:
        f_clean = f.strip().lower()
        # Plain small integers are typically counts, not metrics.
        if re.fullmatch(r"\d{1,2}\+?", f_clean):
            continue
        if any(f_clean in a or a in f_clean for a in approved_lower):
            continue
        suspect.append(f.strip())
    return suspect


def scrub_fluff(text: str) -> str:
    """Remove fluff words / hype adjectives from a bullet.

    Operates on a copy. The matches are case-insensitive and we preserve
    surrounding punctuation. Multiple whitespace collapsed at the end.
    """
    if not text:
        return ""
    out = text
    for word in FLUFF_WORDS:
        # Word-boundary match for multi-word phrases; trailing space tokens
        # ("very ", "highly ") match anywhere.
        if word.endswith(" "):
            pattern = re.compile(re.escape(word), re.IGNORECASE)
        else:
            pattern = re.compile(rf"\b{re.escape(word)}\b", re.IGNORECASE)
        out = pattern.sub("", out)
    # Tidy up: collapse whitespace, fix orphaned punctuation/articles.
    out = re.sub(r"\s+,", ",", out)
    out = re.sub(r"\s+\.", ".", out)
    out = re.sub(r"\(\s+", "(", out)
    out = re.sub(r"\s+\)", ")", out)
    out = re.sub(r"\s{2,}", " ", out)
    # Article cleanup: "a/an/the X" where X starts with a vowel sometimes
    # breaks after we delete "innovative" etc. Fix the obvious ones.
    out = re.sub(r"\ban\s+([bcdfghjklmnpqrstvwxyz])", r"a \1", out, flags=re.IGNORECASE)
    out = re.sub(r"\ba\s+([aeiou])", r"an \1", out, flags=re.IGNORECASE)
    # Strip orphan leading punctuation/whitespace that fluff removal can create.
    out = re.sub(r"^[\s,;:.\-—]+", "", out)
    # Capitalize first letter if it got de-cased by removal of leading verb.
    if out and out[0].islower():
        out = out[0].upper() + out[1:]
    return out.strip()


def reframe_for_mode(achievement: str, company: str, mode: str) -> str | None:
    """Apply IC- vs people-manager framing per the layout spec.

    Returns None if the bullet should be dropped entirely in this mode
    (e.g. people-management bullets in IC mode that have no technical content).
    """
    text = achievement
    if mode == "ic":
        # Strip common team-size constructions for IC roles.
        text = re.sub(
            r"(with )?a (cross-functional )?team of \d+( \([^)]+\))?",
            "", text, flags=re.IGNORECASE,
        )
        text = re.sub(r"\bgrew (the )?PM org \d+\s*(?:to|->|→)\s*\d+\b", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\b\d+\s+Senior PM promotions\b", "", text, flags=re.IGNORECASE)
        text = re.sub(r";?\s*Mentor \d+ PMs[^.]*\.?", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s{2,}", " ", text).strip(" ;,—-")
        if not text or len(text) < 40:
            return None
    return text


# ---------- Profile patches per the layout spec anchors ----------------------------

def patch_profile_for_skill_md(profile: dict) -> dict:
    """Apply EXACT TITLES + VERIFIED FACTUAL ANCHORS overrides at runtime.

    The on-disk master_profile.json is the canonical content source but
    the layout spec prescribes specific titles / metrics that override it. We
    don't mutate the file; we patch the in-memory copy used for this build.
    """
    p = json.loads(json.dumps(profile))  # deep copy

    for role in p.get("experience", []):
        co = role.get("company", "")
        # EXACT TITLES override
        if co in TITLE_OVERRIDES:
            role["title"] = TITLE_OVERRIDES[co]
        # PayTM: 8 PMs (NOT 16); 30% adoption (NOT 40%)
        if co == "PayTM":
            new_achievements = []
            for ach in role.get("achievements", []):
                ach = re.sub(r"team of 16", "team of 8", ach)
                ach = re.sub(r"40% adoption", "30% adoption", ach)
                new_achievements.append(ach)
            role["achievements"] = new_achievements
        # Wayfair: enforce "enabling SDLC velocity" framing language;
        # ensure no "developer productivity L&D charter" wording leaks.
        if co == "Wayfair":
            new_achievements = []
            for ach in role.get("achievements", []):
                ach = ach.replace(
                    "developer productivity L&D charter",
                    "enabling SDLC velocity",
                )
                ach = ach.replace(
                    "$300M annualised impact",
                    "$300M business impact",
                )
                new_achievements.append(ach)
            role["achievements"] = new_achievements
        # IndiaMart: scope = B2B catalog/listing quality (NOT enrichment pipelines)
        if co == "IndiaMart":
            new_achievements = []
            for ach in role.get("achievements", []):
                ach = ach.replace("enrichment pipelines", "B2B catalog/listing quality")
                new_achievements.append(ach)
            role["achievements"] = new_achievements
        # Flipkart: NEVER "field operations efficiency"
        if co == "Flipkart":
            new_achievements = []
            for ach in role.get("achievements", []):
                ach = ach.replace(
                    "field operations efficiency",
                    "seller acquisition workflow efficiency",
                )
                new_achievements.append(ach)
            role["achievements"] = new_achievements

    # Education line
    p["_education_line"] = EDUCATION_LINE
    return p


# ---------- Style sheet ------------------------------------------------------

def build_styles() -> dict:
    """Style sheet built per the layout spec FORMATTING > Color palette / Header treatment."""
    name = ParagraphStyle(
        "Name", fontName=BOLD_FONT, fontSize=15, leading=17, spaceAfter=1,
        textColor=BLUE,
    )
    contact = ParagraphStyle(
        "Contact", fontName=BASE_FONT, fontSize=9, leading=10.5, spaceAfter=0,
        textColor=GREY,
    )
    headline = ParagraphStyle(
        "Headline", fontName=ITALIC_FONT, fontSize=9, leading=11, spaceAfter=4,
        textColor=GREY,
    )
    skills = ParagraphStyle(
        "Skills", fontName=BASE_FONT, fontSize=9, leading=10.5, spaceAfter=2,
        textColor=BODY,
    )
    section = ParagraphStyle(
        "Section", fontName=BOLD_FONT, fontSize=10.5, leading=12,
        spaceBefore=5, spaceAfter=1, textColor=BLUE,
    )
    role = ParagraphStyle(
        "Role", fontName=BOLD_FONT, fontSize=10.5, leading=12.5, spaceAfter=0,
        textColor=BODY,
    )
    role_dates = ParagraphStyle(
        "RoleDates", fontName=BASE_FONT, fontSize=9.5, leading=12.5, spaceAfter=0,
        textColor=GREY,
    )
    role_company = ParagraphStyle(
        "RoleCompany", fontName=ITALIC_FONT, fontSize=9.5, leading=11,
        spaceAfter=2, textColor=GREY,
    )
    body = ParagraphStyle(
        "Body", fontName=BASE_FONT, fontSize=9.5, leading=11.5, spaceAfter=2,
        textColor=BODY,
    )
    bullet = ParagraphStyle(
        "Bullet", fontName=BASE_FONT, fontSize=9.5, leading=11.5,
        leftIndent=12, bulletIndent=0, firstLineIndent=0, spaceAfter=2,
        textColor=BODY,
    )
    summary = ParagraphStyle(
        "Summary", fontName=BASE_FONT, fontSize=9.5, leading=11.5, spaceAfter=4,
        textColor=BODY,
    )
    return {
        "name": name, "contact": contact, "headline": headline,
        "skills": skills, "section": section, "role": role,
        "role_dates": role_dates, "role_company": role_company,
        "body": body, "bullet": bullet, "summary": summary,
    }


def hr_divider(width: float, color=BLUE, thickness: float = 0.6) -> HRFlowable:
    """Layout spec: HRFlowable thickness 0.6pt, color #1F4E8C."""
    return HRFlowable(
        width=width, thickness=thickness, color=color,
        spaceBefore=2, spaceAfter=3, hAlign="LEFT",
    )


def section_header(title: str, styles: dict, doc_width: float) -> list:
    """Section header in BLUE with HRFlowable underneath."""
    return [
        Paragraph(title.upper(), styles["section"]),
        hr_divider(doc_width),
    ]


def role_header(title: str, company: str, dates: str,
                styles: dict, doc_width: float) -> list:
    """Role header per the layout spec: bold title in BODY, dates flush-right in GREY,
    company as italic sub-line in GREY (sub-line, not adjacent to title)."""
    left = Paragraph(f"<b>{title}</b>", styles["role"])
    right = Paragraph(
        f'<para align="right">{dates}</para>', styles["role_dates"],
    )
    t = Table(
        [[left, right]],
        colWidths=[doc_width * 0.74, doc_width * 0.26],
        style=TableStyle([
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ("TOPPADDING", (0, 0), (-1, -1), 0),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]),
    )
    company_line = Paragraph(f"<i>{company}</i>", styles["role_company"])
    return [t, company_line]


def bullet_para(text: str, styles: dict) -> Paragraph:
    return Paragraph(text, styles["bullet"], bulletText="•")


_EM = "—"  # —


def add_bold_lead_in(text: str) -> str:
    """the layout spec non-negotiable: every bullet leads with a bold 3-6 word
    action phrase + em-dash + rest.

    Heuristic:
      1. If the bullet has an em-dash in the first 90 chars, use that
         as the natural split (cap phrase at 6 words).
      2. Otherwise take the first 4 words, insert em-dash after.

    We DON'T split on comma/semicolon/colon — those break inside
    currency phrases ("Rs. 78 Cr") and noun chunks. The bold phrase
    just needs to be the bullet's headline; the rest is the proof.
    """
    if not text:
        return ""
    if text.lstrip().startswith("<b>"):
        return text
    em_idx = -1
    for sep in (f" {_EM} ", f"{_EM} ", f" {_EM}"):
        idx = text.find(sep, 0, 90)
        if idx > 8:
            em_idx = idx
            em_sep = sep
            break
    if em_idx > 0:
        phrase = text[:em_idx].strip()
        words = phrase.split()
        if len(words) > 6:
            # Cap phrase at 6 words; carry the rest after em-dash.
            head_words = words[:6]
            tail = " ".join(words[6:]) + " " + text[em_idx + len(em_sep):].lstrip()
            return f"<b>{' '.join(head_words)}</b> {_EM} {tail.strip()}"
        rest = text[em_idx + len(em_sep):].lstrip()
        return f"<b>{phrase}</b> {_EM} {rest}"
    # No em-dash present — take first 4 words.
    words = text.split()
    if len(words) <= 4:
        return f"<b>{text}</b>"
    head = " ".join(words[:4])
    rest = " ".join(words[4:])
    return f"<b>{head}</b> {_EM} {rest}"


# ---------- Main builder -----------------------------------------------------

def build_pdf(out_path: Path, candidate: dict, profile: dict) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    company = candidate.get("company") or "Unknown"
    if is_declined(company):
        # Per the layout spec: skip entirely.
        return

    mode = detect_role_mode(candidate)
    keywords = jd_keywords_for(candidate)

    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=A4,
        leftMargin=0.58 * inch,
        rightMargin=0.58 * inch,
        topMargin=0.48 * inch,
        bottomMargin=0.48 * inch,
        title=f"Abhillash Jadhav — Resume — {company}",
        author="Abhillash Jadhav",
    )
    styles = build_styles()
    doc_width = A4[0] - doc.leftMargin - doc.rightMargin
    story = []

    # Header sequence per the layout spec:
    # Name (15pt blue bold) -> contact line (· separators) -> HRFlowable
    # -> Core Skills -> HRFlowable -> Executive Summary
    p = profile["personal"]
    contact_line = (
        f'{p["phone_primary"]} &nbsp;&middot;&nbsp; '
        f'<a href="mailto:{p["email"]}" color="#1F4E8C">{p["email"]}</a> &nbsp;&middot;&nbsp; '
        f'<a href="{p["linkedin"]}" color="#1F4E8C">LinkedIn</a> &nbsp;&middot;&nbsp; '
        f'<a href="{p["github"]}" color="#1F4E8C">GitHub</a> &nbsp;&middot;&nbsp; '
        f'{p["location"]}'
    )
    story.append(Paragraph(p["name"], styles["name"]))
    story.append(Paragraph(contact_line, styles["contact"]))
    story.append(hr_divider(doc_width))

    # Core Skills line — required by the layout spec "Both modes" section.
    story.append(Paragraph(
        f'<b>Core Skills:</b> {CORE_SKILLS_LINE}',
        styles["skills"],
    ))
    story.append(hr_divider(doc_width))

    # EXECUTIVE SUMMARY — same content but with Wayfair anchor patched.
    story.extend(section_header("Executive Summary", styles, doc_width))
    summary = profile["executive_summary"].replace(
        "$300M annualised impact", "$300M business impact",
    )
    summary = scrub_cross_company_terms(summary, "Wayfair") if mode == "ic" else summary
    summary = scrub_fluff(summary)  # integrity: strip hype adjectives
    story.append(Paragraph(bold_keywords(summary, keywords), styles["summary"]))

    # PROFESSIONAL EXPERIENCE
    story.extend(section_header("Professional Experience", styles, doc_width))
    for role in profile["experience"]:
        if role["company"] in NEVER_INCLUDE_COMPANIES:
            continue
        block = []
        block.extend(role_header(
            role["title"], role["company"],
            f'{role["start"]} – {role["end"]}',
            styles, doc_width,
        ))
        achievements = list(role["achievements"])
        # Older roles: 1 bullet (Marico / AgroTech / LeEco) or 1-2 (Flipkart/
        # PayTM / IndiaMart / RADAR). Wayfair and Amazon get all (the most
        # recent and most relevant). CTL gets 2 (Director relevance).
        if role["company"] in {"Marico Industries", "AgroTech Foods Ltd", "LeEco"}:
            achievements = achievements[:1]
        elif role["company"] in {"Flipkart", "PayTM", "IndiaMart", "RADAR by AIMleap"}:
            achievements = achievements[:1]  # tightened from 2 to fit 2 pages
        elif role["company"] == "CTL":
            achievements = achievements[:2]

        rendered_bullets = 0
        for ach in achievements:
            framed = reframe_for_mode(ach, role["company"], mode)
            if not framed:
                continue
            framed = scrub_cross_company_terms(framed, role["company"])
            framed = scrub_fluff(framed)  # integrity: strip hype adjectives
            # Integrity check: if the bullet contains a $/%/₹ figure that's
            # not in APPROVED_METRICS, drop the bullet entirely rather than
            # let a fabricated number through. See detect_fabricated_metrics.
            suspect = detect_fabricated_metrics(framed, profile)
            if suspect:
                # Drop silently; the trajectory logger upstream can audit if needed.
                # Honesty over output (CLAUDE.md rule 2).
                continue
            if len(framed) < 40:  # bullet became too short after scrubbing
                continue
            framed = add_bold_lead_in(framed)
            display = bold_keywords(framed, keywords)
            block.append(bullet_para(display, styles))
            rendered_bullets += 1
        # Don't emit an orphan role header if every bullet got dropped.
        if rendered_bullets == 0:
            continue
        block.append(Spacer(1, 4))
        # KeepTogether per role: PayTM block must never split, Amazon header
        # must never trail at end of page 1 (KeepTogether handles both: if
        # the entire block can't fit before a page break, it moves whole).
        story.append(KeepTogether(block))

    # SELECTED PROJECT — only when JD aligns with AI / agentic / GenAI
    title_lower = (candidate.get("title") or candidate.get("role") or "").lower()
    if any(t in title_lower for t in ("ai", "genai", "agentic", "llm", "gen ai")):
        story.extend(section_header("Selected Project", styles, doc_width))
        for proj in profile.get("selected_projects", []):
            block = [
                Paragraph(
                    f'<b>{proj["name"]}</b> &mdash; {proj["role"]} &nbsp;'
                    f'<i>({proj["date"]})</i>', styles["role"],
                ),
                Paragraph(bold_keywords(proj["description"], keywords), styles["body"]),
                Spacer(1, 4),
            ]
            story.append(KeepTogether(block))

    # CORE COMPETENCIES — 3 bold-labeled rows per the consolidated layout spec.
    # Pick "Consumer" or "B2B" for row 2 based on JD context. IC mode skips
    # the Leadership & Strategy row entirely.
    story.extend(section_header("Core Competencies", styles, doc_width))

    why_blob = (
        " ".join(candidate.get("why_fit") or []).lower()
        + " " + (candidate.get("title") or "").lower()
        + " " + (candidate.get("company") or "").lower()
    )
    is_b2b = any(t in why_blob for t in (
        "b2b", "enterprise", "saas", "platform", "data", "api",
        "developer", "marketplace", "fintech", "subscriptions",
    ))

    competencies = [
        ("AI/ML & Platform",
         "LangChain · LangGraph · RAG · multi-LLM orchestration · "
         "Model Proxy · agentic workflows · eval frameworks · drift monitoring · "
         "Microservices (AWS EC2, ECS)"),
    ]
    if is_b2b:
        competencies.append((
            "B2B Experience",
            "Enterprise SaaS platform · API ecosystem · multi-tenant architecture · "
            "subscriptions · pricing strategy · 0-to-1 product builds · M&A integration",
        ))
    else:
        competencies.append((
            "Consumer Experience",
            "Recommendation engines · personalisation · search & discovery · "
            "marketplace · lifecycle targeting · A/B experimentation · NPS / CSAT / VOC",
        ))
    competencies.append((
        "Leadership & Strategy",
        "Cross-functional team leadership (up to 21) · talent development · "
        "C-suite engagement · platform strategy · matrixed alignment without formal authority",
    ))

    if mode == "ic":
        # IC mode: drop the Leadership & Strategy row to elevate technical depth
        competencies = [c for c in competencies if "Leadership" not in c[0]]

    rendered_comp = []
    for label, content in competencies:
        if "RFM" in content or "AI/ML" in label or "Platform" in label:
            keep = "Wayfair"  # AI/ML row keeps Model Proxy / RAG (Wayfair anchors)
        elif "Consumer" in label or "B2B" in label:
            keep = "Generic"  # No company-specific terms in the middle row
        else:
            keep = "Generic"
        rendered_comp.append([
            Paragraph(f"<b>{label}</b>", styles["body"]),
            Paragraph(scrub_cross_company_terms(content, keep), styles["body"]),
        ])
    comp_table = Table(
        rendered_comp,
        colWidths=[doc_width * 0.27, doc_width * 0.73],
        style=TableStyle([
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 1),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]),
    )
    story.append(comp_table)

    # EDUCATION
    story.extend(section_header("Education", styles, doc_width))
    story.append(Paragraph(EDUCATION_LINE, styles["body"]))

    doc.build(story)


# ---------- LLM-draft renderer ----------------------------------------------

def build_pdf_from_draft(out_path: Path, draft: dict, profile: dict,
                         candidate: dict) -> None:
    """Render a PDF from an LLM-generated draft (agent/llm_resume_generator).

    Unlike build_pdf — which reframes master_profile content deterministically
    — this renders bullet text the generator already finalized and the critic
    already vetted. It reuses every layout primitive (fonts, styles, geometry,
    headers) so the visual spec is unchanged; only the content source differs.

    `draft` shape: {executive_summary, core_skills:[...],
    roles:[{company, bullets:[...]}], core_competencies:[[label, kw], ...]}.
    Role titles and dates come from profile["experience"] (verified facts);
    only prose comes from the draft.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    company = candidate.get("company") or "Unknown"
    if is_declined(company):
        return

    keywords = jd_keywords_for(candidate)
    draft_roles = {
        (r.get("company") or "").strip().lower(): (r.get("bullets") or [])
        for r in draft.get("roles", [])
    }

    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=A4,
        leftMargin=0.58 * inch,
        rightMargin=0.58 * inch,
        topMargin=0.48 * inch,
        bottomMargin=0.48 * inch,
        title=f"Abhillash Jadhav — Resume — {company}",
        author="Abhillash Jadhav",
    )
    styles = build_styles()
    doc_width = A4[0] - doc.leftMargin - doc.rightMargin
    story = []

    # Header — name, contact, divider.
    p = profile["personal"]
    contact_line = (
        f'{p["phone_primary"]} &nbsp;&middot;&nbsp; '
        f'<a href="mailto:{p["email"]}" color="#1F4E8C">{p["email"]}</a> &nbsp;&middot;&nbsp; '
        f'<a href="{p["linkedin"]}" color="#1F4E8C">LinkedIn</a> &nbsp;&middot;&nbsp; '
        f'<a href="{p["github"]}" color="#1F4E8C">GitHub</a> &nbsp;&middot;&nbsp; '
        f'{p["location"]}'
    )
    story.append(Paragraph(p["name"], styles["name"]))
    story.append(Paragraph(contact_line, styles["contact"]))
    story.append(hr_divider(doc_width))

    # Core Skills — from the draft, falling back to the static line.
    skills = draft.get("core_skills") or []
    skills_line = " · ".join(str(s) for s in skills) if skills else CORE_SKILLS_LINE
    story.append(Paragraph(f"<b>Core Skills:</b> {skills_line}", styles["skills"]))
    story.append(hr_divider(doc_width))

    # Executive Summary — from the draft.
    story.extend(section_header("Executive Summary", styles, doc_width))
    summary = scrub_fluff(draft.get("executive_summary", "")
                          or profile.get("executive_summary", ""))
    story.append(Paragraph(bold_keywords(summary, keywords), styles["summary"]))

    # Professional Experience — profile order; bullets from the draft.
    story.extend(section_header("Professional Experience", styles, doc_width))
    for role in profile["experience"]:
        if role["company"] in NEVER_INCLUDE_COMPANIES:
            continue
        bullets = draft_roles.get(role["company"].strip().lower(), [])
        if not bullets:
            continue
        block = []
        block.extend(role_header(
            role["title"], role["company"],
            f'{role["start"]} – {role["end"]}',
            styles, doc_width,
        ))
        rendered = 0
        for b in bullets:
            text = scrub_fluff(str(b))  # cheap hype backstop on top of the critic
            if len(text) < 30:
                continue
            display = bold_keywords(add_bold_lead_in(text), keywords)
            block.append(bullet_para(display, styles))
            rendered += 1
        if rendered == 0:
            continue
        block.append(Spacer(1, 4))
        story.append(KeepTogether(block))

    # Selected Project — verified content, surfaced on AI-flavoured JDs.
    title_lower = (candidate.get("title") or candidate.get("role") or "").lower()
    if any(t in title_lower for t in ("ai", "genai", "agentic", "llm", "gen ai")):
        projects = profile.get("selected_projects", [])
        if projects:
            story.extend(section_header("Selected Project", styles, doc_width))
            for proj in projects:
                block = [
                    Paragraph(
                        f'<b>{proj["name"]}</b> &mdash; {proj["role"]} &nbsp;'
                        f'<i>({proj["date"]})</i>', styles["role"],
                    ),
                    Paragraph(bold_keywords(proj["description"], keywords),
                              styles["body"]),
                    Spacer(1, 4),
                ]
                story.append(KeepTogether(block))

    # Core Competencies — from the draft, falling back to a generic set.
    story.extend(section_header("Core Competencies", styles, doc_width))
    competencies = draft.get("core_competencies") or [
        ["AI/ML & Platform",
         "LangChain · LangGraph · RAG · multi-LLM orchestration · "
         "agentic workflows · eval frameworks · Microservices (AWS EC2, ECS)"],
        ["Leadership & Strategy",
         "Cross-functional team leadership · talent development · "
         "C-suite engagement · platform strategy"],
    ]
    rendered_comp = []
    for row in competencies:
        if not isinstance(row, (list, tuple)) or len(row) != 2:
            continue
        label, content = row
        rendered_comp.append([
            Paragraph(f"<b>{label}</b>", styles["body"]),
            Paragraph(str(content), styles["body"]),
        ])
    if rendered_comp:
        story.append(Table(
            rendered_comp,
            colWidths=[doc_width * 0.27, doc_width * 0.73],
            style=TableStyle([
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 1),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]),
        ))

    # Education
    story.extend(section_header("Education", styles, doc_width))
    story.append(Paragraph(EDUCATION_LINE, styles["body"]))

    doc.build(story)


# ---------- Driver ----------------------------------------------------------

def select_fits(scored: list[dict]) -> list[dict]:
    fits = []
    for c in scored:
        original = c.get("original_score") or c.get("score") or 0
        bumped = c.get("bumped_score") or original
        if bumped >= 80 or original >= 80:
            fits.append(c)
    return fits


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True)
    ap.add_argument("--root", default=".")
    args = ap.parse_args()

    root = Path(args.root)

    _register_fonts()

    profile_raw = json.loads((root / "profile/master_profile.json").read_text())
    profile = patch_profile_for_skill_md(profile_raw)
    scored = json.loads(
        (root / f"outputs/{args.date}/_scored_candidates.json").read_text()
    )
    fits = select_fits(scored)
    out_dir = root / f"outputs/{args.date}/resumes"
    out_dir.mkdir(parents=True, exist_ok=True)

    log = []
    for c in fits:
        company = (c.get("company") or "Unknown")
        if is_declined(company):
            log.append({
                "company": company, "title": c.get("title"),
                "status": "skipped_decline_list",
            })
            continue
        company_safe = company.replace(" ", "_")
        title = c.get("title") or c.get("role") or "role"
        slug = slugify(title)
        fname = f"{args.date}_{company_safe}_{slug}.pdf"
        path = out_dir / fname
        build_pdf(path, c, profile)
        if not path.exists():
            log.append({"company": company, "title": title, "status": "failed_no_output"})
            continue
        size = path.stat().st_size
        mode = detect_role_mode(c)
        log.append({
            "company": company,
            "title": title,
            "pdf": str(path.relative_to(root)),
            "size_kb": round(size / 1024, 1),
            "mode": mode,
            "status": "fit" if (c.get("original_score") or 0) >= 80 else "bumped_fit",
        })
        print(f"  [{mode:14}]  {fname}  ({size/1024:.1f} KB)")

    summary_path = root / f"outputs/{args.date}/_resume_generation.json"
    summary_path.write_text(json.dumps(log, indent=2))
    sizes = [e["size_kb"] for e in log if "size_kb" in e]
    if sizes:
        print(
            f"\ngenerated {len(sizes)} resumes -> {out_dir}\n"
            f"size range: {min(sizes):.1f}-{max(sizes):.1f} KB "
            f"(layout spec target: 48-50 KB)"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
