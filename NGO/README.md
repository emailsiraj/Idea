# Healthcare Bot - Medicare Plan Scraper & Downloader

A complete automation pipeline for scraping Medicare plan information from Q1Medicare and downloading policy documents.

## Prerequisites

Install required packages:
```bash
pip install -r /workspaces/Idea/healthcare_bot/req.txt
```

Set up your SerpAPI key as an environment variable:
```bash
export MYVAR="your_serpapi_key_here"
```

**Note:** On Windows, use `set MYVAR=your_serpapi_key_here` instead. Or create a `.env` file in the healthcare_bot directory with: `MYVAR=your_serpapi_key_here`

## Setup Steps

### Step 1: Launch Chrome in Debug Mode

Start a Chrome instance with remote debugging enabled so the scraper can connect to it.

**For Windows:**
```cmd
"C:\Program Files\Google\Chrome\Application\chrome.exe" --remote-debugging-port=9222 --user-data-dir="C:\Users\acer\OneDrive\Desktop\delzonic\pupt\chromesession" --incognito "https://q1medicare.com/"
```

**For Linux:**
```bash
google-chrome \
  --remote-debugging-port=9222 \
  --user-data-dir="$PWD/chromesession" \
  --incognito \
  "https://q1medicare.com/"
```

### Step 2: Complete Cloudflare Verification

Once Chrome opens with the Q1Medicare website, you need to manually complete the Cloudflare bot detection by clicking the checkbox. Keep the browser open - the scraper will use this session.

### Step 3: Verify Target URLs

Check which states/counties you want to scrape by reviewing the target URLs:
```
📁 /workspaces/Idea/healthcare_bot/meta/target_url.json
```

Edit this file if you need to add or remove specific state/county combinations.

### Step 4: Scrape HTML Files

Run the HTML scraper (keep the Chrome browser from Step 1 open):
```bash
python /workspaces/Idea/healthcare_bot/poli_nm_scrapper/policy_nm_scrapper_from_q1.py
```

**Output:** HTML files will be saved to `/workspaces/Idea/healthcare_bot/poli_nm_scrapper/`

### Step 5: Parse Plan Names

Extract Medicare plan names from the HTML files:
```bash
python /workspaces/Idea/healthcare_bot/poli_nm_parser/policy_nm_parse.py
```

**Output:** Text files with plan lists for each county:
- `Collin_plans.txt`
- `Dallas_plans.txt`
- `Denton_plans.txt`
- `Tarrant_plans.txt`

### Step 6: Search for Policy PDFs

Search Google for policy PDF links using the SerpAPI:
```bash
python /workspaces/Idea/healthcare_bot/get_plcy_file_link.py
```

**⚠️ Important:** This script is intentionally manual - edit it to specify which state and plan file to process. This prevents unnecessary API quota usage.

**Output:** JSON files with search results saved to your OneDrive Desktop

### Step 7: Analyze & Process Results

Open and run the Jupyter notebook:
```bash
ana.ipynb
```

**What it does:**
- First 2 cells: Generate a consolidated JSON file with all policy URLs
- Use remaining cells to analyze the results

**Output:** Final consolidated policy JSON file

### Step 8: Download PDF Files

Download all policy PDFs:
```bash
python /workspaces/Idea/healthcare_bot/download_urls.py
```

**Output:** PDF files saved to `/workspaces/Idea/healthcare_bot/pdf_file_downloads/`

## Folder Structure

```
healthcare_bot/
├── poli_nm_scrapper/          # HTML files from Q1Medicare
├── poli_nm_parser/            # Extracted plan names (.txt files)
├── pdf_file_downloads/        # Downloaded PDF documents
├── meta/
│   └── target_url.json        # URLs to scrape
├── policy_nm_scrapper_from_q1.py
├── policy_nm_parse.py
├── get_plcy_file_link.py
├── download_urls.py
└── ana.ipynb
```

## Troubleshooting

- **Chrome connection fails:** Ensure Chrome is still running from Step 1
- **No HTML files generated:** Check target URLs in `meta/target_url.json`
- **API rate limit:** Edit `get_plcy_file_link.py` to process fewer plans at once

