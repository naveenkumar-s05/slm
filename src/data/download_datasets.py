"""Download the license-safe public datasets used by the hybrid corpus.

Only small, openly licensed sources are used:
  * mariiazhiv/cybersecurity_qa        (Apache-2.0)
  * Rowden/CybersecurityQAA            (MIT)
  * Ngadou/social-engineering-convo    (Apache-2.0)

Run:  python -m src.data.download_datasets
"""

from __future__ import annotations

import hashlib
import json
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"

SOURCES = [
    {
        "name": "mariiazhiv/cybersecurity_qa",
        "license": "Apache-2.0",
        "url": "https://huggingface.co/datasets/mariiazhiv/cybersecurity_qa/resolve/main/cybersecurity_qa.jsonl",
        "filename": "mariiazhiv_cybersecurity_qa.jsonl",
        "usage": "General cybersecurity instruction/response pairs; cleaned and mapped to the CyberAware schema.",
    },
    {
        "name": "Rowden/CybersecurityQAA",
        "license": "MIT",
        "url": "https://huggingface.co/datasets/Rowden/CybersecurityQAA/resolve/main/cybersecurityQAAdataset.json",
        "filename": "rowden_cybersecurity_qaa.json",
        "usage": "Policy and advice question/answer pairs; answers condensed to awareness guidance.",
    },
    {
        "name": "Ngadou/social-engineering-convo",
        "license": "Apache-2.0",
        "url": "https://huggingface.co/datasets/Ngadou/social-engineering-convo/resolve/main/text.csv",
        "filename": "ngadou_social_engineering_convo.csv",
        "usage": "Scam / not-scam conversations converted into 'is this a scam?' awareness examples.",
    },
]


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, destination: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "cyberaware-slm/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response, destination.open("wb") as out:
        while True:
            chunk = response.read(1024 * 256)
            if not chunk:
                break
            out.write(chunk)


def main(force: bool = False) -> int:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []
    for source in SOURCES:
        destination = RAW_DIR / source["filename"]
        if force or not destination.exists():
            print(f"downloading {source['name']} ...")
            download(source["url"], destination)
        else:
            print(f"using cached {destination.name}")
        record = dict(source)
        record["path"] = str(destination.relative_to(ROOT)).replace("\\", "/")
        record["sha256"] = sha256_of(destination)
        record["bytes"] = destination.stat().st_size
        record["retrieved_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        manifest.append(record)

    manifest_path = RAW_DIR / "SOURCES.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {manifest_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(force="--force" in sys.argv))
