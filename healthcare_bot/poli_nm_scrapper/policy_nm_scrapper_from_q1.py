import requests
import json
from pathlib import Path
from urllib.parse import urlparse, parse_qs
from playwright.sync_api import sync_playwright



# Get WebSocket URL from existing Chrome
info = requests.get("http://localhost:9222/json/version").json()
CDP_WS = info["webSocketDebuggerUrl"]

with open(r'C:\Users\acer\OneDrive\Desktop\SIraj\meta\target_url.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
    TARGET_URLS = data['urls']

# TARGET_URL = "https://q1medicare.com/PartD-SearchMA-Medicare-2026PlanFinder.php?state=TX&countyCode=48085&showCounty=Collin"

def get_html_from_existing_chrome(TAR_URL):
    with sync_playwright() as p:
        print("Connecting to existing Chrome…")
        browser = p.chromium.connect_over_cdp(CDP_WS)

        # Use existing context OR create new
        context = browser.contexts[0] if browser.contexts else browser.new_context()

        # Use existing page OR create new
        page = context.pages[0] if context.pages else context.new_page()

        print("Connected. Navigating…")
        page.goto(TAR_URL, wait_until="domcontentloaded", timeout=0)

        # Optional wait for JS-rendered content
        page.wait_for_timeout(3000)

        print("Saving final HTML…")

        html = page.content()

        # Ensure output folder is the poli_nm_scrapper directory (same folder as this script)
        out_dir = Path(__file__).resolve().parent

        # Prefer explicit showCounty query value as filename; fall back to the raw split
        try:
            qs = parse_qs(urlparse(TAR_URL).query)
            county = qs.get("showCounty", [None])[0]
        except Exception:
            county = None

        if not county:
            county = TAR_URL.split("showCounty=")[-1]

        # sanitize filename minimally
        safe_name = "".join(c for c in county if c.isalnum() or c in (' ', '_', '-')).rstrip()
        if not safe_name:
            safe_name = "output"

        file_path = out_dir / (safe_name + ".html")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(html)

        print(f"✅ Saved: {file_path}")


if __name__ == "__main__":
    for TAR_URL in TARGET_URLS:
        
        get_html_from_existing_chrome(TAR_URL)
