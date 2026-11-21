"""Pin upstream sources (commit SHA, license, last push) via the public GitHub API and record local dependencies.

Writes results/audit/sources.json. Anonymous, read-only requests; no credentials or personal data are sent.
Re-run before any GPU job: if a pinned SHA moved, the run must use the recorded SHA (git checkout <sha>).
"""
import importlib.metadata as md
import json
import os
import ssl
import urllib.request

try:
    import certifi
    CTX = ssl.create_default_context(cafile=certifi.where())
except ImportError:  # pragma: no cover
    CTX = ssl.create_default_context()

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REPOS = {
    "PAFT/OpenAgent": "LAMDA-NeSy/OpenAgent",
    "Fission-GRPO": "zxzadm/Fission-GRPO",
    "Agent-STAR": "WxxShirley/Agent-STAR",
    "CSO": "kiaia/CSO",
    "RFT-vs-SFT (Lai et al.)": "zhhvvv/rft_vs_sft",
    "BFCL (gorilla)": "ShishirPatil/gorilla",
}
PAPERS = {
    "PAFT": "arXiv 2607.01084 (ICML 2026)", "Fission-GRPO": "ACL 2026 long 1880 / arXiv 2601.15625",
    "Agent Error Dataset": "arXiv 2609.40111 v2", "DART": "EMNLP 2026 main 1036", "Agent-STAR": "arXiv 2603.21972 (EMNLP 2026)",
    "CSO": "Findings ACL 2026 1974", "RFT mitigates forgetting": "PMLR v306 lai26h (ICML 2026)",
}


def gh(path):
    req = urllib.request.Request("https://api.github.com/" + path, headers={"User-Agent": "research-audit", "Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(req, timeout=30, context=CTX) as r:
        return json.load(r)


def main():
    out = {"repos": {}, "papers": PAPERS, "local": {}}
    for k, r in REPOS.items():
        try:
            info = gh(f"repos/{r}")
            c = gh(f"repos/{r}/commits?per_page=1")[0]
            out["repos"][k] = {"repo": r, "sha": c["sha"], "commit_date": c["commit"]["committer"]["date"],
                               "license": (info.get("license") or {}).get("spdx_id"), "pushed_at": info.get("pushed_at"),
                               "archived": info.get("archived")}
        except Exception as e:  # noqa: BLE001
            out["repos"][k] = {"repo": r, "error": str(e)[:200]}
    for p in ("bfcl-eval", "torch", "transformers", "peft", "accelerate", "numpy"):
        try:
            out["local"][p] = md.version(p)
        except md.PackageNotFoundError:
            out["local"][p] = None
    out["model"] = {"id": "Qwen/Qwen3-4B-Instruct-2507", "revision": "cdbee75f17c01a7cc42f958dc650907174af0554"}
    od = os.path.join(ROOT, "results", "audit"); os.makedirs(od, exist_ok=True)
    json.dump(out, open(os.path.join(od, "sources.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
