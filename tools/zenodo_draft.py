"""Create (or update) the Zenodo DRAFT for this project. Never publishes: publishing is a separate, explicit step.

    ZENODO_TOKEN=... python tools/zenodo_draft.py create          # new draft + reserved DOI, prints id and DOI
    ZENODO_TOKEN=... python tools/zenodo_draft.py upload <id>     # replace the draft's file with release/*.zip
    ZENODO_TOKEN=... python tools/zenodo_draft.py newversion <published id>  # draft of a new version of a published record
    ZENODO_TOKEN=... python tools/zenodo_draft.py publish <id>    # PERMANENT; only after the authors approve

The token is read from the environment and is never written to disk.
"""
import json
import os
import sys
from pathlib import Path

import requests

API = "https://zenodo.org/api/deposit/depositions"
ROOT = Path(__file__).resolve().parents[1]
DESCRIPTION = (
    "<p>Code, tests, configuration, pre-registrations (with dated amendments), aggregate results and figures, and the "
    "manuscript source for <em>When does per-object arbitration help? Late fusion over frozen camera, LiDAR and radar detectors in simulated and real adverse weather</em>.</p>"
    "<p>A small router (~18k parameters) reads per-object degradation probes and cross-sensor conflict from cached outputs of "
    "frozen detectors and sets the parameters of one interpretable late-fusion rule; a conformal layer attaches a coverage "
    "guarantee. Evaluated on Boreas (with graded weather simulation), CADC (real snow) and MAN TruckScenes (real rain, snow, "
    "fog, night; 4D radar). All negative and refuted pre-registered outcomes are reported in <code>notes/</code>, including the finding that the per-object advantage is not distinguishable from zero when the detectors are fitted in domain.</p>"
    "<p><strong>Not included:</strong> the Boreas, CADC and MAN TruckScenes datasets (obtain them from their owners; "
    "TruckScenes is CC BY-NC-SA 4.0), detector caches, fitted-model pickles and third-party checkpoints. "
    "Every number in the paper is generated from the files in <code>results/</code> by "
    "<code>paper/latex/make_paper_data.py</code>.</p>")
METADATA = {
    "upload_type": "software",
    "title": "Code, pre-registrations and results for: When does per-object arbitration help? Late fusion over frozen camera, LiDAR and radar detectors in simulated and real adverse weather",
    "description": DESCRIPTION,
    "creators": [
        {"name": "Saleem, Hassan Waqas", "affiliation": "School of Electrical Engineering and Computer Science (SEECS), "
                                                        "National University of Sciences and Technology (NUST), Islamabad, Pakistan",
         "orcid": "0009-0003-4461-8005"},
        {"name": "Saleem, Rahat Zeeshan", "affiliation": "NUST Business School (NBS), National University of Sciences and "
                                                         "Technology (NUST), Islamabad, Pakistan"},
    ],
    "keywords": ["sensor fusion", "adverse weather", "3D object detection", "conformal prediction", "autonomous driving",
                 "pre-registration"],
    "license": "mit",
    "access_right": "open",
    "version": "2.0.0",
    "language": "eng",
    "prereserve_doi": True,
}


def hdr():
    return {"Authorization": f"Bearer {os.environ['ZENODO_TOKEN']}",
            "User-Agent": "paper02-release-script/1.0 (author deposit; h.waqas160@gmail.com)"}


def create():
    r = requests.post(API, headers=hdr(), json={"metadata": METADATA}, timeout=60)
    r.raise_for_status()
    d = r.json()
    print("draft id:", d["id"], "| state:", d["state"], "| reserved DOI:", d["metadata"]["prereserve_doi"]["doi"])
    print("private edit page:", d["links"]["html"])


def setmeta(dep_id):
    meta = {k: v for k, v in METADATA.items() if k != "prereserve_doi"}
    r = requests.put(f"{API}/{dep_id}", headers=hdr(), json={"metadata": meta}, timeout=60)
    if not r.ok:
        sys.exit(f"{r.status_code}: {r.text[:800]}")
    d = r.json()
    print("metadata set on draft", d["id"], "| state:", d["state"], "| DOI:", d["metadata"]["prereserve_doi"]["doi"])


def upload(dep_id):
    d = requests.get(f"{API}/{dep_id}", headers=hdr(), timeout=60).json()
    if d["state"] != "unsubmitted":
        sys.exit("refusing: deposition is not an unpublished draft")
    for f in d.get("files", []):
        requests.delete(f"{API}/{dep_id}/files/{f['id']}", headers=hdr(), timeout=60).raise_for_status()
    bucket = d["links"]["bucket"]
    for p in sorted((ROOT / "release").glob("*")):
        with open(p, "rb") as fh:
            requests.put(f"{bucket}/{p.name}", data=fh, headers=hdr(), timeout=1800).raise_for_status()
        print("uploaded", p.name, f"{p.stat().st_size / 2 ** 20:.1f} MB")


def newversion(dep_id):
    r = requests.post(f"{API}/{dep_id}/actions/newversion", headers=hdr(), timeout=120)
    if not r.ok:
        sys.exit(f"{r.status_code}: {r.text[:800]}")
    draft = requests.get(r.json()["links"]["latest_draft"], headers=hdr(), timeout=60).json()
    print("new-version draft id:", draft["id"], "| state:", draft["state"], "| reserved DOI:", draft["metadata"].get("prereserve_doi", {}).get("doi"))


def publish(dep_id):
    r = requests.post(f"{API}/{dep_id}/actions/publish", headers=hdr(), timeout=120)
    r.raise_for_status()
    print("PUBLISHED:", r.json()["doi_url"])


if __name__ == "__main__":
    cmd = sys.argv[1]
    {"create": create, "newversion": lambda: newversion(sys.argv[2]), "meta": lambda: setmeta(sys.argv[2]), "upload": lambda: upload(sys.argv[2]),
     "publish": lambda: publish(sys.argv[2])}[cmd]()
