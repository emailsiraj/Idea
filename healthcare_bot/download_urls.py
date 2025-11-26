#!/usr/bin/env python3
"""
Download content from URLs in a JSON file.
Saves content organized by entry name with retry logic and error handling.
"""

import json
import os
import sys
import time
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError


class URLDownloader:
    def __init__(self, json_file, output_dir="pdf_file_downloads", max_retries=3, timeout=10):
        self.json_file = json_file
        self.output_dir = Path(output_dir)
        self.max_retries = max_retries
        self.timeout = timeout
        self.output_dir.mkdir(exist_ok=True)
        
        self.stats = {
            "total_urls": 0,
            "successful": 0,
            "failed": 0,
            "skipped": 0,
        }
    
    def load_json(self):
        """Load URLs from JSON file."""
        try:
            with open(self.json_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading JSON file: {e}")
            sys.exit(1)
    
    def get_filename_from_url(self, url):
        """Extract filename from URL, or generate one."""
        try:
            parsed = urlparse(url)
            filename = os.path.basename(parsed.path)
            if filename and '.' in filename:
                return filename
        except:
            pass
        return None
    
    def download_url(self, url, save_path):
        """Download URL content with retry logic."""
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        for attempt in range(1, self.max_retries + 1):
            try:
                request = Request(url, headers=headers)
                with urlopen(request, timeout=self.timeout) as response:
                    content = response.read()
                    with open(save_path, 'wb') as f:
                        f.write(content)
                    print(f"  ✓ Downloaded: {url[:60]}...")
                    return True
            except HTTPError as e:
                if e.code == 404:
                    print(f"  ✗ 404 Not Found: {url[:60]}...")
                    return False
                elif attempt < self.max_retries:
                    print(f"  ! HTTP {e.code} (retry {attempt}/{self.max_retries}): {url[:60]}...")
                    time.sleep(1)
                else:
                    print(f"  ✗ HTTP {e.code} after retries: {url[:60]}...")
                    return False
            except URLError as e:
                if attempt < self.max_retries:
                    print(f"  ! Connection error (retry {attempt}/{self.max_retries}): {e.reason}")
                    time.sleep(1)
                else:
                    print(f"  ✗ Connection failed after retries: {e.reason}")
                    return False
            except Exception as e:
                print(f"  ✗ Unexpected error: {str(e)[:60]}...")
                return False
        
        return False
    
    def download_all(self):
        """Download all URLs from JSON file."""
        data = self.load_json()
        
        if not data:
            print("No data found in JSON file.")
            return
        
        print(f"\n📥 Starting downloads to: {self.output_dir.absolute()}\n")
        
        for entry_name, urls in data.items():
            if not isinstance(urls, list):
                continue
            
            if not urls:
                print(f"⊘ {entry_name}: (no URLs)")
                continue
            
            # Create subdirectory for this entry
            entry_dir = self.output_dir / entry_name
            entry_dir.mkdir(exist_ok=True)
            
            print(f"📦 {entry_name}: {len(urls)} URL(s)")
            
            for idx, url in enumerate(urls, 1):
                self.stats["total_urls"] += 1
                
                if not url or not url.startswith(('http://', 'https://')):
                    print(f"  ⊘ [{idx}] Invalid URL: {url[:60]}...")
                    self.stats["skipped"] += 1
                    continue
                
                filename = self.get_filename_from_url(url)
                if not filename:
                    filename = f"file_{idx}"
                
                # Ensure unique filename
                save_path = entry_dir / filename
                counter = 1
                while save_path.exists():
                    name, ext = filename.rsplit('.', 1) if '.' in filename else (filename, '')
                    new_name = f"{name}_{counter}.{ext}" if ext else f"{filename}_{counter}"
                    save_path = entry_dir / new_name
                    counter += 1
                
                if self.download_url(url, save_path):
                    self.stats["successful"] += 1
                else:
                    self.stats["failed"] += 1
            
            print()
        
        self.print_summary()
    
    def print_summary(self):
        """Print download statistics."""
        print("\n" + "="*60)
        print(f"📊 Download Summary")
        print("="*60)
        print(f"Total URLs:     {self.stats['total_urls']}")
        print(f"✓ Successful:   {self.stats['successful']}")
        print(f"✗ Failed:       {self.stats['failed']}")
        print(f"⊘ Skipped:      {self.stats['skipped']}")
        print(f"Output folder:  {self.output_dir.absolute()}")
        print("="*60 + "\n")


if __name__ == '__main__':
    # Configuration
    json_file = 'doc_links.json'  # Change if needed
    output_dir = 'pdf_file_downloads'      # Change if needed
    
    # Validate JSON file exists
    if not os.path.exists(json_file):
        print(f"Error: {json_file} not found in current directory.")
        sys.exit(1)
    
    # Run downloader
    downloader = URLDownloader(json_file, output_dir)
    downloader.download_all()
