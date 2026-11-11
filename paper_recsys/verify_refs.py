#!/usr/bin/env python3
"""Verify refs.bib against Crossref (DOI) and arXiv (eprint) using curl (system SSL).
No Google/DBLP. Prints a comparison table.
"""
import re, json, subprocess, time, html

BIB = "/Users/lvchao0428/project/ownRecBole/RecBole/paper_recsys/refs.bib"

def parse_bib(path):
    txt = open(path, encoding="utf-8").read()
    entries = []
    for m in re.finditer(r"@(\w+)\{([^,]+),(.*?)\n\}", txt, re.S):
        etype, key, body = m.group(1), m.group(2).strip(), m.group(3)
        fields = {}
        for fm in re.finditer(r"(\w+)\s*=\s*\{(.*?)\}\s*,?\s*(?=\n\s*\w+\s*=|\Z)", body, re.S):
            fields[fm.group(1).lower()] = re.sub(r"\s+", " ", fm.group(2)).strip()
        entries.append({"type": etype, "key": key, **fields})
    return entries

def clean(s):
    if not s: return ""
    s = re.sub(r"[{}]", "", s).replace("\\&", "&").replace("--", "-")
    return re.sub(r"\s+", " ", s).strip().lower()

def curl(url):
    try:
        out = subprocess.run(["curl", "-s", "--max-time", "25",
                              "-A", "refcheck/1.0 (mailto:test@example.com)", url],
                             capture_output=True, text=True, timeout=30)
        return out.stdout
    except Exception as e:
        return ""

def crossref(doi):
    raw = curl("https://api.crossref.org/works/" + doi)
    if not raw: return None
    try:
        m = json.loads(raw)["message"]
    except Exception:
        return None
    yr = ""
    for k in ("published-print", "published-online", "issued"):
        dp = m.get(k, {}).get("date-parts")
        if dp and dp[0] and dp[0][0]:
            yr = str(dp[0][0]); break
    authors = []
    for a in m.get("author", []) or []:
        fam = a.get("family", ""); giv = a.get("given", "")
        authors.append((fam + " " + giv).strip())
    return {"title": clean(" ".join(m.get("title", [""]) or [""])),
            "page": clean(m.get("page", "")), "year": yr,
            "authors": authors, "container": " ".join(m.get("container-title", []) or [])}

def arxiv(eid):
    raw = curl("http://export.arxiv.org/api/query?id_list=" + eid)
    titles = re.findall(r"<title>(.*?)</title>", raw, re.S)
    t = clean(html.unescape(titles[1])) if len(titles) > 1 else ""
    return {"title": t}

entries = parse_bib(BIB)
rows = []
issues = []
for e in entries:
    key = e["key"]
    bt, bp, by = clean(e.get("title","")), clean(e.get("pages","")), e.get("year","").strip()
    doi = e.get("doi","").strip()
    jr = (e.get("journal","") + " " + e.get("eprint",""))
    am = re.search(r"(\d{4}\.\d{4,5})", jr)
    src = tstat = pstat = ystat = "-"
    detail = ""
    if doi and not doi.lower().startswith("10.48550"):
        src = "crossref"
        d = crossref(doi)
        if not d:
            detail = "fetch-failed"
        else:
            tstat = "OK" if (bt in d["title"] or d["title"] in bt) else "MISMATCH"
            if bp and d["page"]:
                pstat = "OK" if bp == d["page"] else f"DIFF(bib={bp}|cr={d['page']})"
            if by and d["year"]:
                ystat = "OK" if by == d["year"] else f"DIFF(bib={by}|cr={d['year']})"
    elif am:
        src = "arxiv"
        d = arxiv(am.group(1))
        tstat = "OK" if d["title"] and (bt in d["title"] or d["title"] in bt) else ("MISMATCH" if d["title"] else "fetch-failed")
        detail = "preprint(no pages/year check)"
    else:
        src = "no-doi"
        detail = "manual (stable classic)"
    rows.append((key, src, tstat, ystat, pstat, detail))
    if "MISMATCH" in (tstat, ystat, pstat) or "DIFF" in ystat or "DIFF" in pstat or detail in ("fetch-failed",):
        issues.append(key)
    time.sleep(0.5)

w = max(len(r[0]) for r in rows)
print(f"{'key'.ljust(w)}  src        title     year        pages")
print("-"*90)
for k,s,t,y,p,d in rows:
    line = f"{k.ljust(w)}  {s.ljust(9)}  {t.ljust(8)}  {y.ljust(10)}  {p}"
    if d: line += f"   [{d}]"
    print(line)
print("\n=== ISSUES (need attention) ===")
print(", ".join(issues) if issues else "NONE — all fetched entries match")
