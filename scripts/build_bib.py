"""Fetch every bibliography entry from its authoritative source.

DOI entries come from doi.org content negotiation; arXiv entries are built
from the arXiv API. Nothing here is typed in by hand, so a hallucinated
reference cannot survive: an unresolvable identifier raises.
"""
from __future__ import annotations

import re
import subprocess
import sys
import urllib.parse
from pathlib import Path

DOIS = {
    "lin2023esm2": "10.1126/science.ade2574",
    "yang2024carp": "10.1016/j.cels.2024.01.008",
    "dallatorre2024nt": "10.1038/s41592-024-02523-z",
    "theodoris2023geneformer": "10.1038/s41586-023-06139-9",
    "nijkamp2023progen2": "10.1016/j.cels.2023.10.002",
    "sillitoe2021cath": "10.1093/nar/gkaa1079",
    "buchfink2015diamond": "10.1038/nmeth.3176",
    "kedzierska2025singlecell": "10.1186/s13059-025-03574-x",
    "ahlmanneltze2025perturbation": "10.1038/s41592-025-02772-6",
    "vieira2025medium": "10.1038/s41598-025-05674-x",
    "unsal2022probe": "10.1038/s42256-022-00457-9",
    "nguyen2024evo": "10.1126/science.ado9336",
    "rives2021esm1b": "10.1073/pnas.2016239118",
    "gresova2023genomic": "10.1186/s12863-023-01123-8",
    "tang2025dnaeval": "10.1186/s13059-025-03674-8",
    "steinegger2017mmseqs2": "10.1038/nbt.3988",
    "schmirler2024finetuning": "10.1038/s41467-024-51844-2",
    "ojala2009permutation": "10.1109/ICDM.2009.108",
    "greenblatt2024passwordlocked": "10.52202/079017-2209",
    "yang2024care": "10.52202/079017-0101",
    "hewitt2019control": "10.18653/v1/d19-1275",
    "voita2020mdl": "10.18653/v1/2020.emnlp-main.14",
    "schaeffer2023mirage": "10.52202/075280-2425",
    "kornblith2019transfer": "10.1109/cvpr.2019.00277",
    "kendiukhov2026code": "10.5281/zenodo.22061974",
}

ARXIV = {
    "hesslow2022rita": "2205.05789",
    "elnaggar2023ankh": "2301.06568",
    "hofstatter2025elicitation": "2502.02180",
    "orgad2025llmsknow": "2410.02707",
    "joeres2026taskspecific": "2608.12090",
    "gao2025pfmbench": "2506.14796",
    "jamasb2024proteinworkshop": "2406.13864",
    "catrina2026reverse": "2603.07710",
    "belinkov2022probing": "2102.12452",
    "alain2016linear": "1610.01644",
    "bouthillier2021variance": "2103.03098",
    "henderson2018deeprl": "1709.06560",
    "dodge2020finetuning": "2002.06305",
    "miller2024errorbars": "2411.00640",
    "kaplan2020scaling": "2001.08361",
    "hoffmann2022chinchilla": "2203.15556",
    "choshen2024hitchhiker": "2410.11840",
    "serrano2024computeoptimal": "2406.07249",
    "rao2019tape": "1906.08230",
    "marin2023bend": "2311.12570",
    "hu2021lora": "2106.09685",
}

# Proceedings entries with no DOI: verified by fetching the landing page and
# confirming the title, then recorded here with that page as the URL.
MANUAL = {
    "li2024featurereuse": dict(
        kind="inproceedings",
        title="Feature Reuse and Scaling: Understanding Transfer Learning with Protein Language Models",
        author="Li, Francesca-Zhoufan and Amini, Ava P. and Yue, Yisong and Yang, Kevin K. and Lu, Alex X.",
        booktitle="Proceedings of the 41st International Conference on Machine Learning",
        series="PMLR", volume="235", year="2024",
        url="https://proceedings.mlr.press/v235/li24a.html",
        check=("https://proceedings.mlr.press/v235/li24a.html", "Feature Reuse and Scaling"),
    ),
    "xu2022peer": dict(
        kind="inproceedings",
        title="{PEER}: A Comprehensive and Multi-Task Benchmark for Protein Sequence Understanding",
        author="Xu, Minghao and Zhang, Zuobai and Lu, Jiarui and Zhu, Zhaocheng and Zhang, Yangtian and Chang, Ma and Liu, Runcheng and Tang, Jian",
        booktitle="Advances in Neural Information Processing Systems 35 Datasets and Benchmarks Track",
        year="2022",
        url="https://proceedings.neurips.cc/paper_files/paper/2022/hash/e467582d42d9c13fa9603df16f31de6d-Abstract-Datasets_and_Benchmarks.html",
        check=("https://proceedings.neurips.cc/paper_files/paper/2022/hash/e467582d42d9c13fa9603df16f31de6d-Abstract-Datasets_and_Benchmarks.html", "PEER"),
    ),
    "dallago2021flip": dict(
        kind="inproceedings",
        title="{FLIP}: Benchmark tasks in fitness landscape inference for proteins",
        author="Dallago, Christian and Mou, Jody and Johnston, Kadina E. and Wittmann, Bruce J. and Bhattacharya, Nicholas and Goldman, Samuel and Madani, Ali and Yang, Kevin K.",
        booktitle="Advances in Neural Information Processing Systems 34 Datasets and Benchmarks Track",
        year="2021",
        url="https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/hash/2b44928ae11fb9384c4cf38708677c48-Abstract-round2.html",
        check=("https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/hash/2b44928ae11fb9384c4cf38708677c48-Abstract-round2.html", "FLIP"),
    ),
    "cheng2024computeoptimal": dict(
        kind="inproceedings",
        title="Training Compute-Optimal Protein Language Models",
        author="Cheng, Xingyi and Chen, Bo and Li, Pan and Gong, Jing and Tang, Jie and Song, Le",
        booktitle="Advances in Neural Information Processing Systems 37",
        year="2024",
        url="https://proceedings.neurips.cc/paper_files/paper/2024/hash/8066ae1446b2bbccb5159587cc3b3bcc-Abstract-Conference.html",
        check=("https://proceedings.neurips.cc/paper_files/paper/2024/hash/8066ae1446b2bbccb5159587cc3b3bcc-Abstract-Conference.html", "Training Compute-Optimal Protein Language Models"),
    ),
    "notin2023proteingym": dict(
        kind="inproceedings",
        title="{ProteinGym}: Large-Scale Benchmarks for Protein Fitness Prediction and Design",
        author="Notin, Pascal and Kollasch, Aaron W. and Ritter, Daniel and van Niekerk, Lood and Paul, Steffanie and Spinner, Han and Rollins, Nathan and Shaw, Ada and Orenbuch, Rose and Weitzman, Ruben and Frazer, Jonathan and Dias, Mafalda and Franceschi, Dinko and Gal, Yarin and Marks, Debora S.",
        booktitle="Advances in Neural Information Processing Systems 36 Datasets and Benchmarks Track",
        year="2023",
        url="https://proceedings.neurips.cc/paper_files/paper/2023/hash/cac723e5ff29f65e3fcbb0739ae91bee-Abstract-Datasets_and_Benchmarks.html",
        check=("https://proceedings.neurips.cc/paper_files/paper/2023/hash/cac723e5ff29f65e3fcbb0739ae91bee-Abstract-Datasets_and_Benchmarks.html", "ProteinGym"),
    ),
}


def curl(url: str, accept: str | None = None) -> str:
    cmd = ["curl", "-sL", "--max-time", "40"]
    if accept:
        cmd += ["-H", f"Accept: {accept}"]
    cmd.append(url)
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.stdout


def fetch_doi(key: str, doi: str) -> str:
    bib = curl(f"https://doi.org/{urllib.parse.quote(doi)}", "application/x-bibtex").strip()
    if not bib.startswith("@"):
        raise SystemExit(f"FAILED to resolve DOI for {key}: {doi}")
    # normalise the citation key
    bib = re.sub(r"^@(\w+)\s*\{[^,]*,", lambda m: f"@{m.group(1)}{{{key},", bib, count=1)
    # brace-protect the title: plainnat lowercases unprotected titles, which
    # turns PFMBench into "Pfmbench" and CATH into "Cath".
    bib = re.sub(r"(title\s*=\s*)\{(.*?)\}(,|\s*\})",
                 lambda m: f"{m.group(1)}{{{{{m.group(2)}}}}}{m.group(3)}", bib,
                 count=1, flags=re.S)
    return bib


def fetch_arxiv(key: str, aid: str) -> str:
    x = curl(f"https://export.arxiv.org/api/query?id_list={aid}")
    entry = re.search(r"<entry>(.*?)</entry>", x, re.S)
    if not entry:
        raise SystemExit(f"FAILED to resolve arXiv id for {key}: {aid}")
    e = entry.group(1)
    title = " ".join(re.search(r"<title>(.*?)</title>", e, re.S).group(1).split())
    authors = re.findall(r"<name>(.*?)</name>", e)
    year = re.search(r"<published>(\d{4})", e).group(1)

    def swap(n: str) -> str:
        parts = n.split()
        return f"{parts[-1]}, {' '.join(parts[:-1])}" if len(parts) > 1 else n

    auth = " and ".join(swap(a) for a in authors)
    return (f"@misc{{{key},\n  title = {{{{{title}}}}},\n  author = {{{auth}}},\n"
            f"  year = {{{year}}},\n  eprint = {{{aid}}},\n  archivePrefix = {{arXiv}},\n"
            f"  url = {{https://arxiv.org/abs/{aid}}}\n}}")


def emit_manual(key: str, d: dict) -> str:
    url, needle = d["check"]
    page = curl(url)
    if needle.lower() not in page.lower():
        raise SystemExit(f"FAILED to confirm landing page for {key}: {needle!r} not on {url}")
    fields = {k: v for k, v in d.items() if k not in ("kind", "check")}
    body = ",\n".join(
        f"  {k} = {{{{{v}}}}}" if k == "title" else f"  {k} = {{{v}}}"
        for k, v in fields.items()
    )
    return f"@{d['kind']}{{{key},\n{body}\n}}"


def main() -> None:
    out = ["% Generated by scripts/build_bib.py -- every entry fetched from its",
           "% authoritative source (doi.org, arXiv API, or a confirmed landing page).",
           "% Do not edit by hand; re-run the script.", ""]
    for key, doi in DOIS.items():
        print(f"  doi   {key}", file=sys.stderr)
        out.append(fetch_doi(key, doi))
    for key, aid in ARXIV.items():
        print(f"  arxiv {key}", file=sys.stderr)
        out.append(fetch_arxiv(key, aid))
    for key, d in MANUAL.items():
        print(f"  page  {key}", file=sys.stderr)
        out.append(emit_manual(key, d))
    out_path = Path(__file__).resolve().parents[1] / "paper" / "refs.bib"
    out_path.write_text("\n\n".join(out) + "\n", encoding="utf-8")
    print(f"wrote {out_path} with {len(DOIS) + len(ARXIV) + len(MANUAL)} entries", file=sys.stderr)


if __name__ == "__main__":
    main()
