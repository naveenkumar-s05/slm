"""Build the hybrid CyberAware dataset.

Pipeline:
  1. Download / read three license-safe public sources plus our curated set.
  2. Normalise and clean every record.
  3. Remove invalid, unsafe-instruction and duplicate records.
  4. Map public Q&A into the strict CyberAware response schema.
  5. Split into train / validation / held-out test with a fixed seed.
  6. Write JSONL splits, the combined dataset and a statistics report.

Run:  python -m src.data.build_dataset
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from src.data.curated_data import CURATED
from src.model.prompting import (
    CATEGORIES,
    SYSTEM_PROMPT,
    format_target,
    is_unsafe_request,
)

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
SPLIT_DIR = ROOT / "data" / "splits"
SEED = 42
DEDUPE_THRESHOLD = 0.85
CROSS_SPLIT_THRESHOLD = 0.80
MAX_PUBLIC_TEST_PER_CATEGORY = 8
MAX_PUBLIC_VAL_PER_CATEGORY = 10

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "than", "so", "to", "of",
    "in", "on", "at", "for", "with", "without", "from", "by", "as", "is", "are",
    "was", "were", "be", "been", "being", "do", "does", "did", "can", "could",
    "should", "would", "will", "may", "might", "must", "i", "you", "he", "she",
    "it", "we", "they", "my", "your", "our", "their", "this", "that", "these",
    "those", "what", "which", "who", "whom", "how", "why", "when", "where",
    "about", "into", "over", "under", "up", "out", "not", "no", "yes", "also",
}

AWARENESS_TERMS = (
    "password", "passphrase", "mfa", "authentication", "phishing", "email",
    "scam", "fraud", "malware", "virus", "ransomware", "trojan", "spyware",
    "backup", "restore", "update", "patch", "device", "phone", "mobile",
    "wifi", "wi-fi", "browser", "website", "link", "download", "privacy",
    "personal", "social media", "account", "login", "suspicious", "employee",
    "staff", "awareness", "protect", "safe", "data", "identity", "hacked",
    "compromise", "incident", "report", "vpn", "cloud", "encryption",
    "credential", "theft", "attacker", "unauthorized",
)

OFFENSIVE_QUESTION_PATTERNS = (
    r"\bhow (?:do i|can i|to|would i) (?:hack|bypass|exploit|crack|steal|ddos)\b",
    r"\b(?:write|create|make|build|develop|code|deploy) (?:a |an |the )?(?:malware|virus|ransomware|trojan|keylogger|botnet|exploit|phishing kit)\b",
    r"\b(?:hack|steal|crack) (?:someone|somebody|a person|another person|an account|a password|wifi|wi-fi)\b",
    r"\b(?:access|break into) (?:someone|somebody|a person|another person)(?:'s)?\b",
    r"\b(?:disable|bypass|evade) (?:the )?(?:antivirus|security|defender|detection|edr)\b",
    r"\b(?:spy|snoop) on\b",
    r"\b(?:ddos|dos) attack\b",
    r"\b(?:reverse shell|sqlmap|metasploit|keylogger code|credential dump)\b",
)

OFFENSIVE_ANSWER_PATTERNS = (
    r"\bhere (?:is|are) (?:how|the steps|a script|the code)\b",
    r"\bstep[- ]by[- ]step\b.*\b(?:hack|exploit|bypass|attack)\b",
    r"\b(?:run|execute) (?:this|the following) (?:command|script|code)\b",
    r"\b(?:reverse shell|keylogger code|phishing kit|credential dump)\b",
    r"\b(?:use|install) (?:metasploit|sqlmap|nmap)\b",
    r"```",
)

CATEGORY_KEYWORDS: dict[str, list[tuple[str, float]]] = {
    "phishing": [
        ("phishing", 4), ("smishing", 4), ("vishing", 3), ("email scam", 3),
        ("suspicious email", 3), ("malicious email", 3), ("fake email", 3),
        ("email attachment", 2), ("email link", 2), ("sender", 1.5),
        ("verify your account", 2.5), ("account will be closed", 3),
        ("clicked a link", 2.5), ("parcel", 1.5), ("delivery fee", 2),
    ],
    "passwords": [
        ("password", 3), ("passphrase", 3), ("mfa", 4), ("multi-factor", 4),
        ("two-factor", 4), ("2fa", 4), ("authenticator", 3), ("password manager", 3),
        ("credential", 2), ("one-time code", 3), ("otp", 2), ("pin", 2),
        ("sign-in", 1), ("account recovery", 2),
    ],
    "malware": [
        ("malware", 4), ("ransomware", 4), ("virus", 3), ("trojan", 3),
        ("spyware", 3), ("keylogger", 3), ("worm", 2), ("infected", 3),
        ("antivirus", 2), ("malicious software", 3), ("malicious file", 2.5),
        ("encrypted my files", 3), ("demands payment", 2),
    ],
    "social-engineering": [
        ("social engineering", 4), ("scam", 3), ("impersonat", 3), ("fraud", 2),
        ("gift card", 3), ("caller", 2), ("phone call", 2), ("boss", 2),
        ("ceo", 2), ("urgent request", 2), ("verify identity", 2),
        ("tailgating", 4), ("pretexting", 3), ("baiting", 3), ("stranger", 1.5),
        ("friend request", 2), ("whatsapp", 1.5),
    ],
    "privacy": [
        ("privacy", 4), ("personal data", 3), ("personal information", 3),
        ("pii", 3), ("data protection", 3), ("gdpr", 3), ("social media", 2.5),
        ("tracking", 2), ("cookies", 2), ("consent", 2), ("data breach", 2.5),
        ("identity theft", 3), ("birth date", 2), ("id photo", 2.5),
        ("share information", 2),
    ],
    "safe-browsing": [
        ("website", 2), ("browser", 3), ("url", 3), ("download", 2),
        ("https", 3), ("extension", 2), ("pop-up", 3), ("popup", 3),
        ("torrent", 2), ("safe browsing", 4), ("domain", 2.5), ("padlock", 3),
        ("scareware", 3), ("unofficial site", 2.5),
    ],
    "updates": [
        ("update", 3), ("patch", 3), ("firmware", 3), ("outdated", 3),
        ("upgrade", 2), ("software version", 2), ("end of life", 2.5),
        ("no longer supported", 2.5),
    ],
    "public-wifi": [
        ("wi-fi", 4), ("wifi", 4), ("public network", 3), ("hotspot", 4),
        ("vpn", 3), ("airport", 2), ("router", 2),
    ],
    "mobile": [
        ("smartphone", 3), ("phone", 2), ("mobile", 3), ("android", 3),
        ("ios", 3), ("sim", 3), ("bluetooth", 2), ("jailbreak", 3),
        ("rooted", 2), ("device", 1), ("app permission", 2.5),
    ],
    "incident-response": [
        ("incident", 3), ("compromised", 4), ("hacked", 4), ("breach", 3),
        ("account takeover", 4), ("suspicious activity", 2.5), ("remote access", 3),
        ("evidence", 3), ("incident response", 4), ("security team", 2),
        ("unauthorized", 2), ("account was hacked", 4), ("preserve", 2),
    ],
    "backup": [
        ("backup", 4), ("back up", 4), ("restore", 3), ("data loss", 3),
        ("3-2-1", 4), ("cloud backup", 4), ("external hard drive", 2.5),
        ("recovery", 2),
    ],
    "general": [
        ("cybersecurity", 2.5), ("security", 1), ("vulnerability", 2.5),
        ("threat", 2), ("awareness", 2), ("risk", 1), ("training", 1.5),
        ("culture", 1.5), ("everyone's responsibility", 2),
    ],
}

CATEGORY_PRIORITY = [
    "incident-response", "phishing", "social-engineering", "malware", "passwords",
    "privacy", "safe-browsing", "updates", "public-wifi", "mobile", "backup", "general",
]

RISK_DEFAULT = {
    "phishing": "medium",
    "passwords": "low",
    "malware": "medium",
    "social-engineering": "medium",
    "privacy": "medium",
    "safe-browsing": "medium",
    "updates": "medium",
    "public-wifi": "medium",
    "mobile": "medium",
    "incident-response": "high",
    "backup": "low",
    "unsafe-request": "high",
    "general": "low",
}

HIGH_RISK_TERMS = (
    "compromised", "hacked", "stolen", "lost", "ransomware", "clicked", "gave",
    "breach", "fraud", "scam", "bank", "money", "gift card", "one-time code",
    "remote access", "infected", "encrypted", "demand", "urgent", "without permission",
    "someone has access", "account takeover", "malicious attachment", "data loss",
)

ACTION_VARIANTS: dict[str, list[str]] = {
    "phishing": [
        "Do not click links or open attachments; report the message to your IT or security team.",
        "Verify the sender through a separate known channel before responding, then report the message.",
        "Report the email as phishing and delete it without opening any links or attachments.",
    ],
    "passwords": [
        "Turn on multi-factor authentication and use a unique password for every account.",
        "Use a reputable password manager to generate a unique password, then enable MFA.",
        "Change the password from a trusted device and enable multi-factor authentication.",
    ],
    "malware": [
        "Disconnect the device from the network and run a full scan with updated security software.",
        "Isolate the device and have your IT or security team inspect it before further use.",
        "Do not open the file; delete it and run a full antivirus scan on the device.",
    ],
    "social-engineering": [
        "Verify the request through an independent, known channel before sharing anything.",
        "Stop the conversation and confirm the person's identity using contact details you already trust.",
        "Do not share codes, passwords, or payment details; report the attempt to the platform.",
    ],
    "privacy": [
        "Review the app or site's privacy settings and share only the minimum information required.",
        "Remove unnecessary personal details from the profile and limit its visibility.",
        "Check the privacy policy and disable optional data collection before continuing.",
    ],
    "safe-browsing": [
        "Close the page, avoid the download, and check the address or source before continuing.",
        "Type the official website address directly instead of following the link.",
        "Do not install anything from the page; run your own security scan if you downloaded a file.",
    ],
    "updates": [
        "Install the update from the official vendor or app store as soon as possible.",
        "Enable automatic updates and apply critical security patches in the next maintenance window.",
        "Update through the official app or website rather than a link in a message.",
    ],
    "public-wifi": [
        "Avoid sensitive logins on public Wi-Fi and use a trusted VPN or mobile hotspot.",
        "Disconnect from the public network and reconnect through a trusted VPN or hotspot.",
        "Use your mobile hotspot for sensitive work and avoid entering credentials on public Wi-Fi.",
    ],
    "mobile": [
        "Review app permissions, keep the operating system updated, and enable device lock and remote wipe.",
        "Install apps only from official stores and revoke permissions the app does not need.",
        "Lock or erase the device remotely, then change the passwords for accounts signed in on it.",
    ],
    "incident-response": [
        "Report the incident to your IT or security team immediately and preserve evidence.",
        "Isolate the affected device from the network and contact the security team before doing anything else.",
        "Change affected credentials, sign out all sessions, and report the incident through the official channel.",
    ],
    "backup": [
        "Verify a recent backup and test that you can restore important files.",
        "Keep at least one backup offline or off-site and confirm the backup job succeeds.",
        "Restore from your latest known-good backup and check that the files open correctly.",
    ],
    "unsafe-request": [
        "Ask for defensive guidance instead, and report harmful requests to the appropriate authority.",
        "Request defensive security guidance or contact your IT team for help.",
        "Stop and ask for legitimate, defensive assistance from a trusted security professional.",
    ],
    "general": [
        "Follow your organization's security policy and ask your IT or security team when unsure.",
        "Report the concern through the official security channel and follow the guidance you receive.",
        "Review your organization's awareness guidance and ask before acting on anything unclear.",
    ],
}

META_SENTENCE_PATTERNS = (
    r"^(?:as an ai|i am an ai|i'm an ai|note that|in conclusion|overall|remember that)\b",
    r"^(?:here are|here is|below are)\b",
)

INVALID_PLACEHOLDERS = (
    "lorem ipsum", "insert ", "todo", "tbd", "n/a", "unknown answer",
)


def normalise_text(value: Any) -> str:
    text = "" if value is None else str(value)
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def normalise_question(value: Any) -> str:
    text = normalise_text(value)
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalise_key(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def token_set(text: str) -> set[str]:
    return {token for token in normalise_key(text).split() if len(token) > 2 and token not in STOPWORDS}


def jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def has_keyword(text: str, keyword: str) -> bool:
    if re.fullmatch(r"[a-z0-9]+", keyword):
        return re.search(rf"\b{re.escape(keyword)}\b", text) is not None
    return keyword in text


def is_awareness_relevant(question: str) -> bool:
    lowered = question.lower()
    return any(has_keyword(lowered, term) for term in AWARENESS_TERMS)


def looks_offensive_question(question: str) -> bool:
    lowered = question.lower()
    return any(re.search(pattern, lowered) for pattern in OFFENSIVE_QUESTION_PATTERNS)


def looks_offensive_answer(answer: str) -> bool:
    lowered = answer.lower()
    return any(re.search(pattern, lowered) for pattern in OFFENSIVE_ANSWER_PATTERNS)


def classify_category(question: str, answer: str) -> tuple[str, dict[str, float]]:
    question_l = question.lower()
    answer_l = answer.lower()
    scores: dict[str, float] = {}
    for category, keywords in CATEGORY_KEYWORDS.items():
        score = 0.0
        for keyword, weight in keywords:
            if has_keyword(question_l, keyword):
                score += weight * 2.0
            if has_keyword(answer_l, keyword):
                score += weight
        if score:
            scores[category] = score

    if not scores:
        return "general", {}

    best_score = max(scores.values())
    for category in CATEGORY_PRIORITY:
        if abs(scores.get(category, -1.0) - best_score) < 1e-9:
            return category, scores
    return max(scores, key=scores.get), scores


def classify_risk(question: str, category: str) -> str:
    if category == "unsafe-request":
        return "high"
    lowered = question.lower()
    if any(term in lowered for term in HIGH_RISK_TERMS):
        return "high"
    return RISK_DEFAULT.get(category, "medium")


def clean_answer(answer: str) -> str:
    text = normalise_text(answer)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"[*_`#>]+", " ", text)
    text = re.sub(r"(?m)^\s*(?:[-•*]|\d+[.)])\s+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"])", text)
    return [part.strip() for part in parts if part.strip()]


def condense_advice(answer: str, max_sentences: int = 3, max_chars: int = 600) -> str:
    text = clean_answer(answer)
    if len(text) < 40:
        return ""
    sentences = []
    for sentence in split_sentences(text):
        if any(re.search(pattern, sentence.lower()) for pattern in META_SENTENCE_PATTERNS):
            continue
        sentences.append(sentence)
        if len(sentences) >= max_sentences:
            break
    advice = " ".join(sentences).strip()
    if len(advice) > max_chars:
        advice = advice[:max_chars].rsplit(" ", 1)[0].rstrip(",;:")
        if not advice.endswith((".", "!", "?")):
            advice += "."
    if len(advice) < 20:
        return ""
    return advice


def choose_action(question: str, category: str, answer: str) -> str:
    variants = ACTION_VARIANTS.get(category, ACTION_VARIANTS["general"])
    digest = hashlib.md5(normalise_key(question).encode("utf-8")).hexdigest()
    return variants[int(digest, 16) % len(variants)]


def make_record(
    *,
    source: str,
    question: Any,
    answer: Any,
    category: str,
    risk: str,
    advice: str,
    action: str,
    split: str | None = None,
) -> dict[str, Any] | None:
    question_text = normalise_question(question)
    advice_text = normalise_text(advice)
    action_text = normalise_question(action)

    if not (10 <= len(question_text) <= 500):
        return None
    if not (20 <= len(advice_text) <= 700):
        return None
    if not (5 <= len(action_text) <= 260):
        return None
    if any(placeholder in (question_text + " " + advice_text).lower() for placeholder in INVALID_PLACEHOLDERS):
        return None
    if category not in CATEGORIES or risk not in {"low", "medium", "high"}:
        return None

    record = {
        "id": f"{source_tag(source)}-{hashlib.sha1(normalise_key(question_text).encode('utf-8')).hexdigest()[:12]}",
        "source": source,
        "question": question_text,
        "category": category,
        "risk": risk,
        "advice": advice_text,
        "action": action_text,
        "target": format_target(
            {"category": category, "risk": risk, "advice": advice_text, "action": action_text}
        ),
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question_text},
            {
                "role": "assistant",
                "content": format_target(
                    {"category": category, "risk": risk, "advice": advice_text, "action": action_text}
                ),
            },
        ],
    }
    if split:
        record["split"] = split
    return record


def source_tag(source: str) -> str:
    tags = {
        "curated": "cur",
        "mariiazhiv/cybersecurity_qa": "mar",
        "Rowden/CybersecurityQAA": "row",
        "Ngadou/social-engineering-convo": "nga",
    }
    return tags.get(source, "src")


def load_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_mariiazhiv() -> tuple[list[dict[str, Any]], Counter]:
    path = RAW_DIR / "mariiazhiv_cybersecurity_qa.jsonl"
    stats: Counter = Counter()
    records: list[dict[str, Any]] = []
    for row in load_jsonl(path):
        stats["raw"] += 1
        question = normalise_question(
            " ".join(
                part for part in [row.get("instruction", ""), row.get("input", "")] if str(part).strip()
            )
        )
        answer = row.get("output", "")
        if not question or not answer:
            stats["missing_field"] += 1
            continue
        if looks_offensive_question(question) or looks_offensive_answer(str(answer)):
            stats["offensive"] += 1
            continue
        if not is_awareness_relevant(question):
            stats["off_topic"] += 1
            continue
        advice = condense_advice(str(answer))
        if not advice:
            stats["too_short_or_unusable"] += 1
            continue
        category, _ = classify_category(question, str(answer))
        risk = classify_risk(question, category)
        action = choose_action(question, category, str(answer))
        record = make_record(
            source="mariiazhiv/cybersecurity_qa",
            question=question,
            answer=answer,
            category=category,
            risk=risk,
            advice=advice,
            action=action,
        )
        if record is None:
            stats["invalid_record"] += 1
        else:
            stats["accepted"] += 1
            records.append(record)
    return records, stats


def load_rowden() -> tuple[list[dict[str, Any]], Counter]:
    path = RAW_DIR / "rowden_cybersecurity_qaa.json"
    stats: Counter = Counter()
    records: list[dict[str, Any]] = []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        for key in ("data", "rows", "examples", "dataset"):
            if isinstance(payload.get(key), list):
                payload = payload[key]
                break
    if not isinstance(payload, list):
        raise ValueError("unexpected Rowden JSON structure")

    for item in payload:
        stats["raw"] += 1
        if not isinstance(item, dict):
            stats["invalid_item"] += 1
            continue
        fields = item.get("vars", item)
        question = normalise_question(fields.get("question", ""))
        answer = fields.get("answer", "")
        if not question or not answer:
            stats["missing_field"] += 1
            continue
        if looks_offensive_question(question) or looks_offensive_answer(str(answer)):
            stats["offensive"] += 1
            continue
        if not is_awareness_relevant(question):
            stats["off_topic"] += 1
            continue
        advice = condense_advice(str(answer))
        if not advice:
            stats["too_short_or_unusable"] += 1
            continue
        category, _ = classify_category(question, str(answer))
        risk = classify_risk(question, category)
        action = choose_action(question, category, str(answer))
        record = make_record(
            source="Rowden/CybersecurityQAA",
            question=question,
            answer=answer,
            category=category,
            risk=risk,
            advice=advice,
            action=action,
        )
        if record is None:
            stats["invalid_record"] += 1
        else:
            stats["accepted"] += 1
            records.append(record)
    return records, stats


def load_ngadou() -> tuple[list[dict[str, Any]], Counter]:
    path = RAW_DIR / "ngadou_social_engineering_convo.csv"
    stats: Counter = Counter()
    records: list[dict[str, Any]] = []
    frame = pd.read_csv(path)
    for _, row in frame.iterrows():
        stats["raw"] += 1
        conversation = normalise_question(row.get("Conversation", ""))
        classification = normalise_question(row.get("Classification", "")).lower()
        reason = normalise_text(row.get("Reason", ""))
        if not conversation or not reason:
            stats["missing_field"] += 1
            continue
        if len(conversation) > 380:
            conversation = conversation[:377].rsplit(" ", 1)[0] + "..."
        question = f'Is this conversation a scam? "{conversation}"'
        is_scam = "scam" in classification and "not" not in classification
        category = "social-engineering"
        risk = "high" if is_scam else "low"
        if is_scam:
            advice = condense_advice(
                reason
                + " Unexpected requests for credentials, payment, or secrecy are strong warning signs."
            )
            action = "Stop the conversation and verify the person's identity through a trusted channel."
        else:
            advice = condense_advice(
                reason
                + " No clear scam indicators are present, but verify unexpected requests before sharing information."
            )
            action = "Stay cautious and verify any unexpected request through a known contact method."
        if not advice:
            stats["too_short_or_unusable"] += 1
            continue
        record = make_record(
            source="Ngadou/social-engineering-convo",
            question=question,
            answer=reason,
            category=category,
            risk=risk,
            advice=advice,
            action=action,
        )
        if record is None:
            stats["invalid_record"] += 1
        else:
            stats["accepted"] += 1
            records.append(record)
    return records, stats


def load_curated() -> tuple[list[dict[str, Any]], Counter]:
    stats: Counter = Counter()
    records: list[dict[str, Any]] = []
    for item in CURATED:
        stats["raw"] += 1
        record = make_record(
            source="curated",
            question=item["question"],
            answer=item["advice"],
            category=item["category"],
            risk=item["risk"],
            advice=item["advice"],
            action=item["action"],
            split=item["split"],
        )
        if record is None:
            stats["invalid_record"] += 1
            continue
        stats["accepted"] += 1
        records.append(record)
    return records, stats


def dedupe(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    kept: list[dict[str, Any]] = []
    kept_tokens: list[set[str]] = []
    kept_keys: set[str] = set()
    duplicates = 0

    for record in records:
        key = normalise_key(record["question"])
        if key in kept_keys:
            duplicates += 1
            continue
        tokens = token_set(record["question"])
        is_duplicate = False
        for index, existing_tokens in enumerate(kept_tokens):
            if abs(len(tokens) - len(existing_tokens)) > max(4, int(0.5 * max(len(tokens), len(existing_tokens)))):
                continue
            if jaccard(tokens, existing_tokens) >= DEDUPE_THRESHOLD:
                is_duplicate = True
                break
        if is_duplicate:
            duplicates += 1
            continue
        kept.append(record)
        kept_tokens.append(tokens)
        kept_keys.add(key)
    return kept, duplicates


def assign_public_splits(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rng = random.Random(SEED)
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_category[record["category"]].append(record)

    for category in sorted(by_category):
        group = by_category[category]
        group.sort(key=lambda item: item["id"])
        rng.shuffle(group)
        total = len(group)
        if total < 6:
            n_test = 0
            n_val = 0
        else:
            n_test = min(MAX_PUBLIC_TEST_PER_CATEGORY, max(1, math.ceil(total * 0.10)))
            n_val = min(MAX_PUBLIC_VAL_PER_CATEGORY, max(1, math.ceil(total * 0.10)))
        if n_test + n_val >= total:
            n_val = max(1, total - n_test - 1)
        for index, record in enumerate(group):
            if index < n_test:
                record["split"] = "test"
            elif index < n_test + n_val:
                record["split"] = "val"
            else:
                record["split"] = "train"
    return records


def prune_cross_split(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    train_val = [record for record in records if record["split"] in {"train", "val"}]
    train_val_tokens = [(record, token_set(record["question"])) for record in train_val]
    kept: list[dict[str, Any]] = []
    dropped = 0
    for record in records:
        if record["split"] != "test":
            kept.append(record)
            continue
        tokens = token_set(record["question"])
        too_similar = any(
            jaccard(tokens, other_tokens) >= CROSS_SPLIT_THRESHOLD
            for _, other_tokens in train_val_tokens
        )
        if too_similar:
            dropped += 1
        else:
            kept.append(record)
    return kept, dropped


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def summarise(records: list[dict[str, Any]], counts: dict[str, Any]) -> dict[str, Any]:
    by_split = Counter(record["split"] for record in records)
    by_source_split = Counter((record["source"], record["split"]) for record in records)
    by_category_split = Counter((record["category"], record["split"]) for record in records)
    advice_lengths = [len(record["advice"]) for record in records]
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": SEED,
        "dedupe_threshold": DEDUPE_THRESHOLD,
        "cross_split_threshold": CROSS_SPLIT_THRESHOLD,
        "counts": counts,
        "totals": {
            "records": len(records),
            "by_split": dict(sorted(by_split.items())),
            "by_source_split": {f"{source}|{split}": count for (source, split), count in sorted(by_source_split.items())},
            "by_category_split": {f"{category}|{split}": count for (category, split), count in sorted(by_category_split.items())},
            "advice_chars": {
                "min": min(advice_lengths) if advice_lengths else 0,
                "max": max(advice_lengths) if advice_lengths else 0,
                "mean": round(sum(advice_lengths) / len(advice_lengths), 1) if advice_lengths else 0,
            },
        },
        "sources": [
            {"name": "curated", "license": "project-authored", "usage": "Hand-authored examples for schema, safety and topic coverage."},
            {"name": "mariiazhiv/cybersecurity_qa", "license": "Apache-2.0"},
            {"name": "Rowden/CybersecurityQAA", "license": "MIT"},
            {"name": "Ngadou/social-engineering-convo", "license": "Apache-2.0"},
        ],
    }


def main() -> int:
    missing = [
        name
        for name in [
            "mariiazhiv_cybersecurity_qa.jsonl",
            "rowden_cybersecurity_qaa.json",
            "ngadou_social_engineering_convo.csv",
        ]
        if not (RAW_DIR / name).exists()
    ]
    if missing:
        print("Missing raw files: " + ", ".join(missing))
        print("Run: python -m src.data.download_datasets")
        return 1

    curated, curated_stats = load_curated()
    mariiazhiv, mariiazhiv_stats = load_mariiazhiv()
    rowden, rowden_stats = load_rowden()
    ngadou, ngadou_stats = load_ngadou()

    combined = curated + mariiazhiv + rowden + ngadou
    combined, duplicates = dedupe(combined)

    curated_records = [record for record in combined if record["source"] == "curated"]
    public_records = [record for record in combined if record["source"] != "curated"]
    public_records = assign_public_splits(public_records)
    combined = curated_records + public_records
    combined, cross_split_dropped = prune_cross_split(combined)

    by_split = defaultdict(list)
    for record in combined:
        by_split[record["split"]].append(record)

    for split_name in ("train", "val", "test"):
        records = sorted(by_split[split_name], key=lambda item: item["id"])
        write_jsonl(SPLIT_DIR / f"{split_name}.jsonl", records)

    write_jsonl(PROCESSED_DIR / "cyberaware_all.jsonl", sorted(combined, key=lambda item: item["id"]))

    counts = {
        "curated": dict(curated_stats),
        "mariiazhiv/cybersecurity_qa": dict(mariiazhiv_stats),
        "Rowden/CybersecurityQAA": dict(rowden_stats),
        "Ngadou/social-engineering-convo": dict(ngadou_stats),
        "dedupe": {"duplicates_removed": duplicates},
        "cross_split": {"test_records_removed": cross_split_dropped},
    }
    stats = summarise(combined, counts)
    stats_path = PROCESSED_DIR / "dataset_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")

    print(f"train: {len(by_split['train'])}  val: {len(by_split['val'])}  test: {len(by_split['test'])}")
    print(f"duplicates removed: {duplicates}  cross-split test removed: {cross_split_dropped}")
    print(f"wrote {stats_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
