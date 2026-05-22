"""
Generate papers_metadata.csv from arxiv_cs_ai_pdfs/ folder.
Fetches full metadata (title, authors, venue, year, url, topics) via arXiv API.
- Queries one paper at a time to avoid rate-limit (HTTP 429)
- Saves progress after each successful fetch so the script can be re-run safely
- Fixes Windows cp1252 console encoding issues
"""

import os, re, csv, time, sys, json
import urllib.request, urllib.parse
import xml.etree.ElementTree as ET

# ── Fix Windows console encoding ─────────────────────────────────────────────
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# ── Config ────────────────────────────────────────────────────────────────────
PDF_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'arxiv_cs_ai_pdfs')
OUT_CSV    = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'papers_metadata.csv')
CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.meta_cache.json')
DELAY      = 4.0   # seconds between arXiv API calls (polite limit)
RETRY_WAIT = 30.0  # seconds to wait after a 429
MAX_RETRY  = 3

NS = 'http://www.w3.org/2005/Atom'  # Atom namespace used by arXiv API

# ── 1. Collect PDF filenames & parse arXiv IDs ────────────────────────────────
pdf_files = sorted(f for f in os.listdir(PDF_DIR) if f.endswith('.pdf'))
print(f"Found {len(pdf_files)} PDF files.")

id_to_filename = {}
ordered_ids    = []
for fname in pdf_files:
    m = re.match(r'^(\d{4}\.\d{4,6})(v\d+)?', fname)
    if m:
        full_id = m.group(1) + (m.group(2) or '')
        id_to_filename[full_id] = fname
        ordered_ids.append(full_id)

print(f"Parsed {len(ordered_ids)} arXiv IDs.\n")

# ── 2. Load cache (so we can resume after interruption) ───────────────────────
cache = {}
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, encoding='utf-8') as f:
        cache = json.load(f)
    print(f"Loaded {len(cache)} cached entries from {CACHE_FILE}\n")

def save_cache():
    with open(CACHE_FILE, 'w', encoding='utf-8') as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

# ── 3. Fetch metadata for one ID via arXiv Atom API ──────────────────────────
def fetch_one(arxiv_id: str) -> dict | None:
    """Query export.arxiv.org and return a metadata dict or None on failure."""
    base_id = re.sub(r'v\d+$', '', arxiv_id)   # strip version for query
    url = (f"https://export.arxiv.org/api/query"
           f"?id_list={urllib.parse.quote(arxiv_id)}&max_results=1")
    for attempt in range(1, MAX_RETRY + 1):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=20) as resp:
                raw = resp.read()
            root = ET.fromstring(raw)
            entries = root.findall(f'{{{NS}}}entry')
            if not entries:
                return None
            e = entries[0]
            def txt(tag):
                node = e.find(f'{{{NS}}}{tag}')
                return (node.text or '').strip() if node is not None else ''

            title   = txt('title').replace('\n', ' ')
            year_raw = txt('published')[:4]
            pdf_url  = ''
            for link in e.findall(f'{{{NS}}}link'):
                if link.attrib.get('title') == 'pdf':
                    pdf_url = link.attrib.get('href', '')
                    break
            authors = '; '.join(
                (a.find(f'{{{NS}}}name').text or '').strip()
                for a in e.findall(f'{{{NS}}}author')
                if a.find(f'{{{NS}}}name') is not None
            )
            categories = [
                c.attrib.get('term', '')
                for c in e.findall('{http://arxiv.org/schemas/atom}primary_category')
            ] + [
                c.attrib.get('term', '')
                for c in e.findall(f'{{{NS}}}category')   # fallback
            ]
            # deduplicate while preserving order
            seen = set(); cats = []
            for c in categories:
                if c and c not in seen:
                    seen.add(c); cats.append(c)

            journal_ref_node = e.find('{http://arxiv.org/schemas/atom}journal_ref')
            venue = ''
            if journal_ref_node is not None and journal_ref_node.text:
                venue = journal_ref_node.text.strip()
            if not venue and cats:
                venue = cats[0]   # fallback to primary category

            return {
                'title':   title,
                'authors': authors,
                'venue':   venue,
                'year':    year_raw,
                'pdf_url': pdf_url,
                'topics':  '; '.join(cats),
            }
        except urllib.error.HTTPError as e_http:
            if e_http.code == 429:
                print(f"  [429] rate-limited — waiting {RETRY_WAIT}s (attempt {attempt}/{MAX_RETRY})")
                time.sleep(RETRY_WAIT)
            else:
                print(f"  [HTTP {e_http.code}] {arxiv_id} — {e_http}")
                return None
        except Exception as exc:
            print(f"  [ERR attempt {attempt}] {arxiv_id}: {exc}")
            time.sleep(5)
    return None

# ── 4. Main loop ──────────────────────────────────────────────────────────────
total = len(ordered_ids)
for i, full_id in enumerate(ordered_ids, 1):
    if full_id in cache:
        print(f"[{i:>3}/{total}] (cached) {full_id}")
        continue

    print(f"[{i:>3}/{total}] Fetching {full_id} ...", end=' ', flush=True)
    result = fetch_one(full_id)
    if result:
        cache[full_id] = result
        save_cache()
        print(f"OK  — {result['title'][:60]}")
    else:
        print("MISS — using filename fallback")
        raw_title = re.sub(r'^\d{4}\.\d+v\d+_', '', id_to_filename[full_id])
        raw_title = raw_title.replace('.pdf', '').replace('_', ' ')
        cache[full_id] = {
            'title':   raw_title,
            'authors': '',
            'venue':   'arXiv',
            'year':    '2026',
            'pdf_url': f'https://arxiv.org/pdf/{full_id}',
            'topics':  'cs.AI',
        }
        save_cache()

    time.sleep(DELAY)

# ── 5. Write CSV ──────────────────────────────────────────────────────────────
FIELDNAMES = ['paper_id', 'title', 'authors', 'venue', 'year',
              'pdf_path', 'pdf_url', 'topics']

rows = []
for full_id in ordered_ids:
    m = cache.get(full_id, {})
    local_path = os.path.join(PDF_DIR, id_to_filename[full_id])
    rows.append({
        'paper_id': full_id,
        'title':    m.get('title', ''),
        'authors':  m.get('authors', ''),
        'venue':    m.get('venue', ''),
        'year':     m.get('year', ''),
        'pdf_path': local_path,
        'pdf_url':  m.get('pdf_url', ''),
        'topics':   m.get('topics', ''),
    })

with open(OUT_CSV, 'w', newline='', encoding='utf-8') as fh:
    writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
    writer.writeheader()
    writer.writerows(rows)

print(f"\nDone!  {len(rows)} rows (+1 header) -> {OUT_CSV}")
