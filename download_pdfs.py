import os
import time
import arxiv
import requests

# 1. Configuration
SEARCH_QUERY = 'cat:cs.AI'  # Computer Science - Artificial Intelligence category
MAX_RESULTS = 250          # Number of PDFs you want to download
DOWNLOAD_DIR = 'arxiv_cs_ai_pdfs'
DELAY_SECONDS = 10          # Courteous delay to avoid being blocked by arXiv

# Create the directory if it doesn't exist
if not os.path.exists(DOWNLOAD_DIR):
    os.makedirs(DOWNLOAD_DIR)
    print(f"Created directory: {DOWNLOAD_DIR}")

# 2. Initialize the arXiv Client
client = arxiv.Client()

# 3. Construct the Search
search = arxiv.Search(
    query=SEARCH_QUERY,
    max_results=MAX_RESULTS,
    sort_by=arxiv.SortCriterion.SubmittedDate,  # Fetches the most current/recent
    sort_order=arxiv.SortOrder.Descending
)

print(f"Fetching metadata for the top {MAX_RESULTS} papers in {SEARCH_QUERY}...")
results = list(client.results(search))
print(f"Found {len(results)} papers. Starting downloads...")

# 4. Iterate and Download
for i, result in enumerate(results, start=1):
    # Sanitize the title to make a valid filename
    clean_title = "".join(c for c in result.title if c.isalnum() or c in (' ', '_', '-')).rstrip()
    # Format: "arXivID_CleanTitle.pdf"
    pdf_filename = f"{result.get_short_id()}_{clean_title[:50]}.pdf"
    pdf_path = os.path.join(DOWNLOAD_DIR, pdf_filename)
    
    # Check if the file already exists to prevent duplicate downloads
    if os.path.exists(pdf_path):
        print(f"[{i}/{MAX_RESULTS}] Skipped (Already exists): {pdf_filename}")
        continue

    try:
        print(f"[{i}/{MAX_RESULTS}] Downloading: {pdf_filename}...")
        
        # Pull the PDF URL
        pdf_url = result.pdf_url
        
        # Download the file via requests
        response = requests.get(pdf_url, stream=True)
        if response.status_code == 200:
            with open(pdf_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"     Successfully saved.")
        else:
            print(f"     Failed to download. HTTP Status: {response.status_code}")
            
        # Crucial step: Sleep to respect arXiv's servers
        time.sleep(DELAY_SECONDS)
        
    except Exception as e:
        print(f"     Error downloading {result.entry_id}: {e}")
        # If blocked or experiencing a network error, back off for longer
        time.sleep(10)

print("Finished downloading!")