import re
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from bs4 import BeautifulSoup

def extract_links_from_json_content(json_content: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Extract link data from content in JSON format.
    Returns a list of dictionaries with url and title.
    """
    links = []
    
    # Handle different JSON structures
    if isinstance(json_content, dict):
        if "optimized_html" in json_content:
            html_content = json_content["optimized_html"]
            try:
                extracted_links = extract_links_from_html(html_content)
                links.extend(extracted_links)
            except Exception as e:
                logging.error(f"Error extracting links from optimized_html: {str(e)}")
        
    return links

def extract_links_from_html(html_content: str) -> List[Dict[str, str]]:
    """
    Extract all links from HTML content using BeautifulSoup.
    Returns a list of dictionaries with url and title.
    """
    links = []
    try:
        soup = BeautifulSoup(html_content, "html.parser")
        a_tags = soup.find_all('a')
        
        for a in a_tags:
            url = a.get('href', '')
            if url and not url.startswith('#') and not url.startswith('javascript:'):
                title = a.get_text(strip=True) or a.get('title', '')
                links.append({
                    'url': url,
                    'title': title
                })
    except Exception as e:
        logging.error(f"Error parsing HTML to extract links: {str(e)}")
        
    return links

def count_internal_links(html_content: str, site_base_path: str = "") -> Tuple[int, List[str]]:
    """
    Count internal links within HTML content that match the site_base_path.
    Returns a tuple of (count, list of internal link URLs).
    
    Args:
        html_content: HTML content to analyze
        site_base_path: Base path to consider for internal links (e.g. "https://example.com/")
        
    Returns:
        Tuple containing (number of internal links, list of internal link URLs)
    """
    if not html_content:
        return 0, []
        
    internal_urls = []
    try:
        soup = BeautifulSoup(html_content, "html.parser")
        a_tags = soup.find_all('a')
        
        for a in a_tags:
            url = a.get('href', '')
            if url and not url.startswith('#') and not url.startswith('javascript:'):
                # Check if this is an internal link
                if site_base_path and (
                    url.startswith(site_base_path) or 
                    (url.startswith('/') and not url.startswith('//'))
                ):
                    internal_urls.append(url)
                    
    except Exception as e:
        logging.error(f"Error counting internal links: {str(e)}")
        
    return len(internal_urls), internal_urls

def validate_internal_links(links_data: List[Dict[str, Any]], site_base_path: str) -> List[Dict[str, Any]]:
    """
    Validate a list of internal links against a site base path.
    Returns the list with validation info added.
    
    Args:
        links_data: List of link data dictionaries
        site_base_path: Base URL of the site
        
    Returns:
        List of links with validation status added
    """
    validated_links = []
    
    for link in links_data:
        url = link.get('url', '')
        is_internal = False
        is_relative = False
        
        if url.startswith('/') and not url.startswith('//'):
            is_relative = True
            is_internal = True
        elif site_base_path and url.startswith(site_base_path):
            is_internal = True
            
        validated_link = link.copy()
        validated_link['is_internal'] = is_internal
        validated_link['is_relative'] = is_relative
        
        validated_links.append(validated_link)
        
    return validated_links
