from bs4 import BeautifulSoup
import os
from pathlib import Path

# Source directory containing HTML files
# Get the parent directory of the script location
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
SOURCE_DIR = os.path.join(parent_dir, "poli_nm_scrapper")
OUTPUT_DIR = script_dir  # Save output files in script directory

# Create output directory if needed
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Find all HTML files in source directory
html_files = list(Path(SOURCE_DIR).glob("*.html")) + list(Path(SOURCE_DIR).glob("*.htm"))

if not html_files:
    print(f"No HTML files found in {SOURCE_DIR}")
    exit(1)

print(f"Found {len(html_files)} HTML file(s)\n")

total_plans = 0

for html_file in sorted(html_files):
    print(f"Processing: {html_file.name}")
    
    try:
        with open(html_file, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f.read(), "html.parser")
        
        plan_links = []
        
        for a in soup.find_all("a", href=True):
            href = a["href"]
            title = a.get("title", "")
        
            # Condition 1 → Medicare plan link
            if "MedicareAdvantage-2026C-MedicareHealthPlanBenefits.php" not in href:
                continue
        
            # Condition 2 → Title must contain specific phrase
            if "View Enrollment Options" not in title:
                continue
        
            # Extract text as plan name
            name = a.get_text(strip=True)
            plan_links.append(name)
        
        # Generate output filename based on input filename
        output_filename = html_file.stem + "_plans.txt"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        
        # Save results
        with open(output_path, "w", encoding="utf-8") as f:
            for name in plan_links:
                f.write(f"{name}\n")
        
        print(f"  ✓ Extracted {len(plan_links)} plans → {output_filename}\n")
        total_plans += len(plan_links)
    
    except Exception as e:
        print(f"  ✗ Error processing {html_file.name}: {e}\n")

print(f"{'='*50}")
print(f"✓ Complete! Total plans extracted: {total_plans}")
