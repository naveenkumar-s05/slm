# Dataset Report

## 1. Objective

Build a small, high-quality cybersecurity **awareness** dataset that teaches a
1.5B model to answer in a fixed four-line schema, covers 13 categories, and includes
enough safety examples to support a deterministic refusal layer.

The dataset is defensive. It does not contain operational instructions for hacking,
malware creation, credential theft, or evasion. Public rows containing offensive
instructional content were removed during filtering.

## 2. Sources and licences

| Source | Licence | Raw | Accepted | Notes |
|---|---|---:|---:|---|
| Project-curated | Project-authored | 82 | 82 | Schema, safety, and rare-category coverage |
| `mariiazhiv/cybersecurity_qa` | Apache-2.0 | 502 | 144 | General awareness QA |
| `Rowden/CybersecurityQAA` | MIT | 1,563 | 612 | Larger QA corpus after relevance filtering |
| `Ngadou/social-engineering-convo` | Apache-2.0 | 10 | 10 | Social-engineering conversations |

Provenance is stored in `data/raw/SOURCES.json`.

## 3. Cleaning and filtering

1. **Relevance filter:** drop rows that are not about defensive awareness (for example,
   exploit development, certification trivia, or unrelated IT administration).
2. **Safety filter:** drop offensive instructional rows and rows that cannot be
   answered defensively.
3. **Normalisation:** map each row to exactly:
   `Category`, `Risk`, `Advice`, `Action`.
4. **Field bounds:** advice 20–700 characters, action 5–260 characters.
5. **Deduplication:** remove near-duplicates at cosine threshold 0.85.
6. **Split:** fixed seed 42, stratified by category.
7. **Leakage check:** remove test rows whose similarity to a train/validation row is
   at least 0.80.
8. **Validation:** verify schema, allowed labels, counts, and split disjointness.

Filtering removed **1,301 off-topic rows**, **6 offensive rows**, **15 duplicates**, and
**3 cross-split near-duplicates**.

## 4. Final splits

| Split | Records | Purpose |
|---|---:|---|
| Train | 669 | Stage-1 adaptation |
| Validation | 77 | Loss monitoring and checkpoint selection |
| Test | 84 | Final held-out evaluation only |

Balanced training file: **444 records** (`data/splits/train_balanced.jsonl`), used for
stage two. It down-samples dominant classes while retaining rare classes.

## 5. Category distribution

### Test set

| Category | n | Category | n |
|---|---:|---|---:|
| phishing | 11 | incident-response | 11 |
| passwords | 6 | backup | 4 |
| malware | 8 | unsafe-request | 2 |
| social-engineering | 6 | general | 9 |
| privacy | 9 | updates | 5 |
| safe-browsing | 3 | public-wifi | 4 |
| mobile | 6 | | |

### Balanced training set

| Category | n | Category | n |
|---|---:|---|---:|
| phishing | 46 | incident-response | 42 |
| passwords | 42 | backup | 25 |
| malware | 42 | unsafe-request | 12 |
| social-engineering | 28 | general | 42 |
| privacy | 42 | updates | 30 |
| safe-browsing | 26 | public-wifi | 27 |
| mobile | 40 | | |

The balanced set deliberately over-represents rare classes relative to their natural
frequency. This improves coverage but is not a production prevalence estimate.

## 6. Label definitions

| Category | Intended scope |
|---|---|
| `phishing` | Email, SMS, voice, and messaging scams |
| `passwords` | Passwords, MFA, account authentication |
| `malware` | Viruses, ransomware, spyware, malicious downloads |
| `social-engineering` | Manipulation, pretexting, impersonation |
| `privacy` | Personal data, tracking, oversharing, permissions |
| `safe-browsing` | Browsers, extensions, downloads, suspicious sites |
| `updates` | Patching, update hygiene, end-of-life software |
| `public-wifi` | Hotspots, captive portals, untrusted networks |
| `mobile` | Phones, tablets, mobile apps, device locking |
| `incident-response` | What to do after a suspected compromise |
| `backup` | Backup strategy, recovery, ransomware resilience |
| `unsafe-request` | Harmful requests refused with defensive redirection |
| `general` | General awareness and cross-cutting questions |

Risk is one of `low`, `medium`, `high`.

## 7. Quality controls

- Every target parses against the schema; validated in `src/data/validate_data.py`.
- No exact or near-duplicate questions across train/validation/test.
- All 13 categories appear in the test set.
- Safety examples exist in all splits.
- The test set was never used for training or hyperparameter selection.
- Raw public data is retained separately from processed splits for auditability.

Validation result: `data/processed/validation_report.json` reports `"ok": true`.

## 8. Limitations

- 84 test rows give a clear baseline/adaptation signal but wide confidence intervals.
- Label boundaries overlap (public Wi-Fi vs privacy vs safe browsing), which limits
  achievable category accuracy.
- Public QA sources vary in style and quality; curated examples partially compensate.
- The dataset is English-only.
- The dataset is for awareness education, not incident response or legal advice.
