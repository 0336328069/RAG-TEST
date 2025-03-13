import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from typing import List, Dict, Set
import html2text
import time
import random

class WebScraper:
    def __init__(self, base_url: str, max_pages: int = 50):
        """
        Initialize the web scraper.
        
        Args:
            base_url: The main URL of the website to scrape
            max_pages: Maximum number of pages to scrape
        """
        self.base_url = base_url
        self.max_pages = max_pages
        self.visited_urls: Set[str] = set()
        self.h2t = html2text.HTML2Text()
        self.h2t.ignore_links = False
        self.h2t.ignore_images = True
        
        # Common browser headers
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
        
        # Create a session to maintain cookies
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        
    def is_valid_url(self, url: str) -> bool:
        """Check if URL belongs to the same domain as base_url and is valid."""
        try:
            base_domain = urlparse(self.base_url).netloc
            url_domain = urlparse(url).netloc
            
            # Check if URL is valid and belongs to same domain
            return (
                base_domain == url_domain and
                not url.endswith(('.pdf', '.jpg', '.png', '.gif', '.css', '.js')) and
                '#' not in url and
                'mailto:' not in url
            )
        except:
            return False
    
    def extract_text_and_metadata(self, url: str, html_content: str) -> Dict:
        """Extract clean text and metadata from HTML content."""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 'footer', 'iframe', 'header', 'aside']):
                element.decompose()
            
            # Get title
            title = soup.title.string if soup.title else ''
            
            # Try different content selectors based on common website structures
            content = None
            for selector in ['main', 'article', '#content', '.content', '[role="main"]']:
                content = soup.select_one(selector)
                if content and len(content.get_text(strip=True)) > 100:
                    break
            
            if not content:
                content = soup.find('body')
            
            if content:
                # Convert HTML to clean text
                text = self.h2t.handle(str(content))
                # Clean up the text
                text = ' '.join(line.strip() for line in text.split('\n') if line.strip())
            else:
                return None
            
            # Only return if we have meaningful content
            if len(text.strip()) > 100:  # Minimum content length
                return {
                    'text': text.strip(),
                    'metadata': {
                        'source': url,
                        'title': title.strip() if title else '',
                        'type': 'webpage'
                    }
                }
            return None
            
        except Exception as e:
            print(f"Error extracting content from {url}: {str(e)}")
            return None
    
    def scrape_page(self, url: str) -> tuple[List[str], Dict]:
        """Scrape a single page and return found links and content."""
        try:
            # Add random delay between requests (1-3 seconds)
            time.sleep(random.uniform(1, 3))
            
            # Make request with session
            response = self.session.get(
                url,
                timeout=15,
                allow_redirects=True
            )
            response.raise_for_status()
            
            # Ensure we're dealing with HTML content
            content_type = response.headers.get('content-type', '').lower()
            if 'text/html' not in content_type:
                return [], None
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find all links
            links = []
            for a_tag in soup.find_all('a', href=True):
                href = a_tag['href']
                full_url = urljoin(url, href)
                if self.is_valid_url(full_url) and full_url not in self.visited_urls:
                    links.append(full_url)
            
            # Extract content
            document = self.extract_text_and_metadata(url, response.text)
            
            return links, document
            
        except Exception as e:
            print(f"Error scraping {url}: {str(e)}")
            return [], None
    
    def scrape_website(self) -> List[Dict]:
        """
        Scrape only the specified URL.
        
        Returns:
            List of documents with text content and metadata
        """
        documents = []
        
        try:
            print(f"Scraping URL: {self.base_url}")
            # Only scrape the base_url
            _, document = self.scrape_page(self.base_url)
            
            if document:
                print(f"Found content: {document['metadata']['title']}")
                documents.append(document)
                print("Content preview:")
                print("-" * 50)
                print(document['text'][:500] + "...")  # Show first 500 characters
                print("-" * 50)
            else:
                print("No content found at the specified URL")
            
        except Exception as e:
            print(f"Error scraping {self.base_url}: {str(e)}")
            
        return documents 