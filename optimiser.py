import streamlit as st
import os
import json
import html
import re
import time
import requests
import hashlib
import logging
from io import StringIO
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple

try:
    from bs4 import BeautifulSoup
except ImportError:
    st.warning("BeautifulSoup is not installed. Installing a lightweight version...")
    try:
        import pip
        pip.main(['install', 'beautifulsoup4'])
        from bs4 import BeautifulSoup
    except:
        st.error("Failed to install BeautifulSoup. Falling back to basic chunking.")
        BeautifulSoup = None

# Local imports
try:
    from link_validator import count_internal_links, extract_links_from_json_content
except ImportError:
    st.warning("Link validator module not found. Internal link functionality will be limited.")
    
    # Fallback implementations if import fails
    def count_internal_links(html_content, site_base_path=""):
        return 0, []
    
    def extract_links_from_json_content(json_content):
        return []

# --- Constants for Link Database ---
DB_DIR = "link_databases"
MAX_PER_KEYWORD = 5  # Max recent articles to keep per primary keyword
MAX_SITE_ENTRIES = 40  # Max total recent articles to keep per site

# Site-specific base paths (used to standardize URLs)
SITE_BASE_PATHS = {
    "ICOBENCH": "https://icobench.com/", 
    "BITCOINIST": "https://bitcoinist.com/", 
    "CRYPTODNES": "https://cryptodnes.bg/"
}

# WordPress API endpoints - update with your site configurations
WP_API_ENDPOINTS = {
    "ICOBENCH": {
        "api_url": "https://icobench.com/wp-json/wp/v2/posts",
        "username": "",  # Set via environment variable WP_ICOBENCH_USER
        "app_password": ""  # Set via environment variable WP_ICOBENCH_PASSWORD
    },
    "BITCOINIST": {
        "api_url": "https://bitcoinist.com/wp-json/wp/v2/posts",
        "username": "",  # Set via environment variable WP_BITCOINIST_USER
        "app_password": ""  # Set via environment variable WP_BITCOINIST_PASSWORD
    },
    "CRYPTODNES": {
        "api_url": "https://cryptodnes.bg/wp-json/wp/v2/posts",
        "username": "",  # Set via environment variable WP_CRYPTODNES_USER
        "app_password": ""  # Set via environment variable WP_CRYPTODNES_PASSWORD
    }
}

# --- Configure logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def init_openrouter_client():
    """Initialize OpenRouter connection parameters"""
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        raise ValueError("OpenRouter API key not found")
    
    return {
        'api_key': api_key,
        'default_model': "anthropic/claude-sonnet-4",
        'api_url': "https://openrouter.ai/api/v1/chat/completions"
    }

def update_link_database(
    site_name: str,
    article_main_title: str,
    article_meta_desc: str,
    seo_slug: str,
    primary_keyword: str,
    wp_post_url: str = ""
):
    """
    Updates the link database for the given site with the new article's details.

    Args:
        site_name: Name of the site (e.g., "ICOBENCH", "BITCOINIST", "CRYPTODNES").
        article_main_title: The main title of the article.
        article_meta_desc: Meta description.
        seo_slug: SEO-friendly slug for the URL.
        primary_keyword: Primary keyword for the article.
        wp_post_url: WordPress post URL (if available).
    """
    if not site_name:
        logging.error("[LinkDB] Site name is required to update link database.")
        return

    if not os.path.exists(DB_DIR):
        try:
            os.makedirs(DB_DIR)
            logging.info(f"[LinkDB] Created directory: {DB_DIR}")
        except OSError as e:
            logging.error(f"[LinkDB] Error creating directory {DB_DIR}: {e}")
            return

    db_filename = f"{site_name.lower().replace(' ', '_')}_links.json"
    db_path = os.path.join(DB_DIR, db_filename)

    if not seo_slug:
        logging.warning("[LinkDB] Could not update internal link database: Article slug missing.")
        return

    # Extract just the slug from seo_slug or wp_post_url
    slug = seo_slug.strip('/') if seo_slug else wp_post_url.split('/')[-1].strip('/')
    
    # Create standardized URL format according to site type
    if not slug:
        logging.warning("[LinkDB] Cannot update link database: URL slug is missing.")
        st.warning("Could not update internal link database: URL slug missing.")
        return
    
    # Format the URL according to site requirements
    site_key = site_name.upper()
    
    # ICOBench and CryptoNews should have /news/ prefix
    if site_key in ["ICOBENCH", "CRYPTONEWS"]:
        relative_path = f"/news/{slug}/"
    else:
        # Other sites just need the slug with leading slash
        relative_path = f"/{slug}/"
    
    logging.info(f"[LinkDB] Standardized URL path for {site_key}: {relative_path}")

    if not relative_path:
        logging.warning("[LinkDB] Cannot update link database: Relative path is missing.")
        st.warning("Could not update internal link database: Path missing.")
        return

    new_entry = {
        "url": relative_path, # Standardized URL path format
        "title": article_main_title,
        "metaDescription": article_meta_desc,
        "primaryKeyword": primary_keyword,
        "site_name": site_name,
        "publishedTimestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S%z")
    }

    database: List[Dict[str, Any]] = []
    if os.path.exists(db_path):
        try:
            with open(db_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, list):
                database = data
            elif isinstance(data, dict) and 'links' in data and isinstance(data['links'], list):
                database = data['links']
                logging.info(f"[LinkDB] Loaded database with 'links' key structure. Found {len(database)} entries.")
            else:
                logging.warning(f"[LinkDB] {db_path} had unexpected structure. Resetting.")
        except json.JSONDecodeError:
            logging.warning(f"[LinkDB] {db_path} was corrupted or not valid JSON. Starting fresh.")
        except Exception as e:
            logging.error(f"[LinkDB] Error loading {db_path}: {e}")
            st.error(f"Error loading link database {db_path}: {e}")
            return

    # Remove any existing entry with the same URL to avoid duplicates
    database = [entry for entry in database if entry.get("url") != new_entry["url"]]
    database.append(new_entry)

    try:
        # First sort all entries by timestamp (newest first)
        database.sort(key=lambda x: x.get("publishedTimestamp", ""), reverse=True)
        
        # Group articles by primary keyword and keep only the 5 most recent per keyword
        keyword_groups = {}
        for entry in database:
            keyword = entry.get("primaryKeyword", "").lower()
            if keyword not in keyword_groups:
                keyword_groups[keyword] = []
            keyword_groups[keyword].append(entry)
        
        final_database = []
        for keyword_entries in keyword_groups.values():
            final_database.extend(keyword_entries[:MAX_PER_KEYWORD])
        
        # Sort all entries by timestamp again
        final_database.sort(key=lambda x: x.get("publishedTimestamp", ""), reverse=True)
        
        # Apply the global MAX_SITE_ENTRIES limit
        database = final_database[:MAX_SITE_ENTRIES]
        
    except Exception as e:
        logging.error(f"[LinkDB] Error processing database, possibly due to malformed data: {e}")
        # Fall back to the original behavior if there's an error
        database = database[:MAX_SITE_ENTRIES]

    try:
        with open(db_path, 'w', encoding='utf-8') as f:
            json.dump({"links": database}, f, ensure_ascii=False, indent=2)
        logging.info(f"[LinkDB] Link database for {site_name} updated with '{new_entry['title']}'. Count: {len(database)}")
        logging.info(f"[LinkDB] New entry URL structure: {new_entry['url']}")
        logging.info(f"[LinkDB] seo_slug used: '{seo_slug[:30] if seo_slug else 'None'}'...")
    except Exception as e:
        logging.error(f"[LinkDB] Error writing to link database {db_path}: {e}")
        st.error(f"Error writing to link database {db_path}: {e}")


def get_relevant_internal_links(primary_keyword: str, site_name: str, max_links: int = 3) -> List[Dict[str, Any]]:
    """
    Retrieves relevant internal links from the link database based on primary keyword.
    
    Args:
        primary_keyword: The primary keyword to match against
        site_name: The site name to get links from
        max_links: Maximum number of links to return
        
    Returns:
        List of link entries matching the primary keyword
    """
    if not primary_keyword or not site_name:
        return []
        
    db_filename = f"{site_name.lower().replace(' ', '_')}_links.json"
    db_path = os.path.join(DB_DIR, db_filename)
    
    if not os.path.exists(db_path):
        logging.warning(f"[LinkDB] No link database found for site {site_name}")
        return []
        
    try:
        with open(db_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        database = []
        if isinstance(data, list):
            database = data
        elif isinstance(data, dict) and 'links' in data and isinstance(data['links'], list):
            database = data['links']
        else:
            logging.warning(f"[LinkDB] Unexpected structure in {db_path}")
            return []
            
        # First, look for exact primary keyword match
        exact_matches = []
        for entry in database:
            if entry.get("primaryKeyword", "").lower() == primary_keyword.lower():
                exact_matches.append(entry)
                
        # If we have enough exact matches, return those
        if len(exact_matches) >= max_links:
            return exact_matches[:max_links]
            
        # Otherwise look for partial matches to fill the remaining slots
        partial_matches = []
        remaining_slots = max_links - len(exact_matches)
        
        for entry in database:
            if entry not in exact_matches:  # Avoid duplicates
                entry_keyword = entry.get("primaryKeyword", "").lower()
                if (primary_keyword.lower() in entry_keyword or 
                    entry_keyword in primary_keyword.lower()):
                    partial_matches.append(entry)
                    if len(partial_matches) >= remaining_slots:
                        break
                        
        # Combine exact and partial matches
        result = exact_matches + partial_matches
        return result[:max_links]  # Limit to max_links
        
    except Exception as e:
        logging.error(f"[LinkDB] Error retrieving internal links: {e}")
        return []

def extract_json_safely(resp_text):
    """
    Attempts to extract a JSON object from the response text.
    Returns the parsed JSON object or None if extraction fails.
    """
    try:
        json_match = re.search(r'\{.*\}', resp_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
        else:
            start_index = resp_text.find('{')
            end_index = resp_text.rfind('}')
            if start_index != -1 and end_index != -1 and end_index > start_index:
                json_str = resp_text[start_index:end_index + 1]
            else:
                return None
        clean_json_str = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]', '', json_str)
        return json.loads(clean_json_str, strict=False)
    except Exception as e:
        st.error(f"JSON extraction error: {str(e)}")
        return None

def _optimize_single(text: str, primary_keyword: str = "", secondary_keywords: list = None, max_retries=3, simplified_mode=False, internal_links: list = None, site_name: str = None) -> dict:
    """Optimize a single chunk of Thai HTML content using Claude via OpenRouter with retries."""
    secondary_keywords = secondary_keywords or []
    internal_links = internal_links or []
    
    # Format internal links for prompt if available
    internal_links_instruction = ""
    if internal_links and len(internal_links) > 0:
        internal_links_json = json.dumps(internal_links, ensure_ascii=False)[:1500]  # Limit size
        
        internal_links_instruction = f"""
8. Internal Link Integration:
        Insert {min(len(internal_links), 3)} internal links from the provided list as naturally as possible. Follow these rules:
        - Place links in relevant sections where they naturally fit with the content topic
        - Use the article title as anchor text or a relevant portion of it
        - DO NOT place links in headings or image captions
        - DO NOT modify the URLs in any way - use them exactly as provided
        - Format links as standard HTML: <a href="[url-from-json]">[title]</a>
        
        Available internal links (JSON format):
        {internal_links_json}
        """
    
    if simplified_mode:
        # Use a simpler prompt with fewer requirements for problematic chunks
        prompt = f"""You are a Thai SEO specialist focusing on HTML content. Optimize this Thai HTML content by following these critical requirements:

1. Never modify HTML tags or attributes - only optimize the visible Thai text
2. Keep all technical terms and brand names as they are
3. Preserve all URLs, variables, and code exactly as they are
4. Return ONLY a valid JSON object with this exact structure:
{{
  "optimized_html": "HTML content with optimized Thai text",
  "titles": ["Any title found"],
  "meta_descriptions": ["Any meta description found"],
  "alt_text": "Any alt text found",
  "wordpress_slug": "thai-version-of-title"
}}

Content to optimize:
{text}"""
    else:
        # Create the primary keyword instruction separately to avoid nested f-strings
        if primary_keyword:
            primary_keyword_instruction = f"""Ensure that the primary keyword '{primary_keyword}' is integrated naturally throughout the ENTIRE content in the following way:
    - Title (1x)
    - First paragraph (1x)
    - Every major section of content (at least once per section)
    - In at least 2-3 H2 or H3 headings
    - In the conclusion paragraph (1x)
    - Meta description (1x)
    - CRITICAL: The primary keyword MUST appear AT LEAST 5-7 times across the entire content
    - When appropriate, use both exact match and semantic variations of the keyword
    IMPORTANT: DO NOT concentrate keyword usage only at the beginning and end - distribute evenly throughout all sections."""
        else:
            primary_keyword_instruction = "No primary keyword provided."
        
        # Create secondary keyword instructions separately
        secondary_keywords_str = ", ".join(secondary_keywords) if secondary_keywords else "None provided"
        
        if secondary_keywords:
            secondary_keyword_placement = """IMPORTANT: You MUST include EACH secondary keyword at least once in the optimized content. Place secondary keywords strategically in:
   - At least 2-3 H2 or H3 headings
   - Within paragraphs where they fit naturally
   - In the FAQ section questions and answers (if present)"""
            
            priority_keywords = ", ".join(secondary_keywords[:5]) if len(secondary_keywords) > 5 else ", ".join(secondary_keywords)
            priority_instruction = f"Priority secondary keywords (include these first):\n   {priority_keywords}"
            
            limit_instruction = "Limit each secondary keyword to maximum 2 mentions in the entire content."
        else:
            secondary_keyword_placement = "No secondary keywords provided."
            priority_instruction = ""
            limit_instruction = ""
        
        # Build the full prompt without nested f-strings
        prompt = f"""You are a Senior Thai SEO specialist focusing on crypto content. Optimize this Thai HTML content for SEO by following these strict requirements:

1. Preserve HTML Structure
   Keep all HTML tags, attributes, and inline styles completely unchanged. Do not modify the HTML in any way except to optimize the visible Thai text.

2. Retain Technical Content
   Do not modify any technical elements such as URLs, placeholders (e.g., [cur_year], {{{{variable}}}}, and shortcodes (e.g., [su_note], [toc]). They must remain exactly as they are.

3. Keep Entity Names (people, places, organisation), Brands, coin names, and Technical Terms in English.

4. Optimize Content Without Changing Meaning
   You must maintain the meaning and intent of the original Thai content. Your goal is to optimize the text for keywords while keeping the original meaning intact. Do not add new information that wasn't in the original content.

5. Ensure Proper Thai Spacing and Punctuation
   Optimize the content while strictly following Thai orthographic and typographic conventions. Make sure no extra spaces or punctuation errors are introduced.

6. Keyword Integration:
    Primary Keyword: {primary_keyword if primary_keyword else "None provided"}
    {primary_keyword_instruction}
    
   Secondary Keywords: {secondary_keywords_str}
   {secondary_keyword_placement}
   {priority_instruction}
   
   {limit_instruction}

7. VERY IMPORTANT: This is chunk of a larger HTML document. Optimize ONLY what's provided. Don't try to complete or start tags that seem incomplete - they will be joined with other chunks.{internal_links_instruction}

Return exactly a valid JSON object matching the following schema without any additional text:
{{
  "optimized_html": "STRING",
  "titles": ["STRING"],
  "meta_descriptions": ["STRING"],
  "alt_text": "STRING",
  "wordpress_slug": "STRING"
}}

For meta_descriptions, create 3 distinct meta descriptions that:
- Each contains the primary keyword once
- Each contains 2-3 different secondary keywords
- Are between 150-160 characters in length
- Have different sentence structures and focuses

The wordpress_slug should be a Thai-language URL-friendly version that:
- Closely matches high-volume search terms in your market
- Is concise (3-5 words maximum)
- Focuses on the main topic of the content

Content to optimize:
{text}

Return ONLY the JSON object, with no extra text or commentary."""

    
    client_config = init_openrouter_client()
    
    for attempt in range(max_retries):
        try:
            response = requests.post(
                url=client_config['api_url'],
                headers={
                    "Authorization": f"Bearer {client_config['api_key']}",
                    "Content-Type": "application/json"
                },
                data=json.dumps({
                    "model": client_config['default_model'],
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                }),
                timeout=240
            )
            
            response.raise_for_status()
            resp_data = response.json()
            
            if response.status_code == 429:
                wait_time = int(response.headers.get('Retry-After', 60))
                st.warning(f"Rate limited. Waiting {wait_time} seconds before retrying...")
                time.sleep(wait_time)
                continue
            
            if not resp_data or 'choices' not in resp_data or not resp_data['choices']:
                st.error(f"Invalid response format. Attempt {attempt+1}/{max_retries}")
                if attempt == max_retries - 1:
                    return None
                time.sleep(2)
                continue
                
            resp_text = resp_data['choices'][0]['message']['content']
            
        except requests.exceptions.RequestException as e:
            st.error(f"API Request failed: {str(e)} - Attempt {attempt + 1}/{max_retries}")
            if attempt == max_retries - 1:
                return None
            time.sleep(2)
            continue

        if not resp_text:
            st.error(f"Empty response from Claude. Attempt {attempt+1}/{max_retries}")
            if attempt == max_retries - 1:
                return None
            time.sleep(2)
            continue

        result = extract_json_safely(resp_text)
        if result is None:
            st.error(f"Failed to extract JSON on attempt {attempt+1}/{max_retries}. Preview: {resp_text[:300]}")
            if attempt == max_retries - 1:
                return None
            time.sleep(2)
            continue

        # Update key name from translated_html to optimized_html
        if 'translated_html' in result and 'optimized_html' not in result:
            result['optimized_html'] = result.pop('translated_html')

        required = ['optimized_html', 'titles', 'meta_descriptions', 'alt_text']
        optional = ['wordpress_slug']
        
        missing = [field for field in required if field not in result]
        if missing:
            st.error(f"Missing required fields: {missing} Attempt {attempt+1}/{max_retries}")
            if attempt == max_retries - 1:
                return None
            time.sleep(2)
            continue
            
        # Add wordpress_slug if missing
        if 'wordpress_slug' not in result and result.get('titles') and result['titles']:
            # Create a simple slug from the first title
            from re import sub
            title = result['titles'][0]
            result['wordpress_slug'] = sub(r'[^\w\s-]', '', title.lower())
            result['wordpress_slug'] = sub(r'[\s-]+', '-', result['wordpress_slug'])

        result['optimized_html'] = html.unescape(result['optimized_html'])
        return result
    return None

def optimize_chunk_with_fallback(chunk, primary_keyword="", secondary_keywords=None, max_level=3, site_name=None, internal_links=None):
    """
    Attempts to optimize a chunk using _optimize_single.
    If optimization fails and max_level is not exceeded, the chunk is split into halves recursively.
    Returns a dictionary with combined optimized_html and aggregated SEO elements or None if optimization fails.
    """
    secondary_keywords = secondary_keywords or []
    internal_links = internal_links or []
    result = _optimize_single(chunk, primary_keyword, secondary_keywords, internal_links=internal_links, site_name=site_name)
    if result is not None:
        return result
    else:
        if max_level <= 0 or len(chunk) < 1000:
            return None
        mid = len(chunk) // 2
        chunk1 = chunk[:mid]
        chunk2 = chunk[mid:]
        result1 = optimize_chunk_with_fallback(chunk1, primary_keyword, secondary_keywords, max_level-1, site_name=site_name)
        result2 = optimize_chunk_with_fallback(chunk2, "", secondary_keywords, max_level-1, site_name=site_name)
        if result1 is None or result2 is None:
            return None
        optimized_html = result1['optimized_html'] + result2['optimized_html']
        titles = result1['titles'] + result2['titles']
        meta_descriptions = result1['meta_descriptions'] + result2['meta_descriptions']
        alt_text = result1['alt_text'] if result1['alt_text'].strip() else result2['alt_text']
        wordpress_slug = result1.get('wordpress_slug', '') if result1.get('wordpress_slug', '').strip() else result2.get('wordpress_slug', '')
        return {
            'optimized_html': optimized_html,
            'titles': titles,
            'meta_descriptions': meta_descriptions,
            'alt_text': alt_text,
            'wordpress_slug': wordpress_slug
        }

def smart_chunk_html(html_content: str, max_length: int) -> list:
    """
    Intelligently split HTML content into semantically meaningful chunks not exceeding max_length.
    Tries to split at major section boundaries when possible.
    Uses BeautifulSoup if available for better parsing.
    """
    if not BeautifulSoup:
        return [html_content[i:i+max_length] for i in range(0, len(html_content), max_length)]
    
    soup = BeautifulSoup(html_content, "html.parser")
    container = soup.body if soup.body else soup
    section_tags = ['section', 'article', 'div', 'main', 'header', 'footer', 'nav']
    
    major_sections = []
    for tag_name in section_tags:
        sections = container.find_all(tag_name, recursive=False)
        if sections and len(sections) > 1:
            major_sections = sections
            break
    
    if major_sections:
        chunks = []
        current_chunk = StringIO()
        current_size = 0
        
        for section in major_sections:
            section_str = str(section)
            section_len = len(section_str)
            
            if section_len > max_length:
                if current_size > 0:
                    chunks.append(current_chunk.getvalue())
                    current_chunk = StringIO()
                    current_size = 0
                sub_chunks = smart_chunk_html(section_str, max_length)
                chunks.extend(sub_chunks)
            elif current_size + section_len > max_length and current_size > 0:
                chunks.append(current_chunk.getvalue())
                current_chunk = StringIO()
                current_chunk.write(section_str)
                current_size = section_len
            else:
                current_chunk.write(section_str)
                current_size += section_len
        
        if current_size > 0:
            chunks.append(current_chunk.getvalue())
        
        return chunks

    chunks = []
    current_chunk = StringIO()
    current_size = 0
    for element in container.children:
        element_str = str(element)
        element_len = len(element_str)
        if element_str.strip() == '':
            continue
        if element_len > max_length:
            if current_size > 0:
                chunks.append(current_chunk.getvalue())
                current_chunk = StringIO()
                current_size = 0
            if isinstance(element, str):
                for i in range(0, element_len, max_length):
                    chunks.append(element_str[i:i+max_length])
            else:
                sub_chunks = smart_chunk_html(element_str, max_length)
                chunks.extend(sub_chunks)
        elif current_size + element_len > max_length and current_size > 0:
            chunks.append(current_chunk.getvalue())
            current_chunk = StringIO()
            current_chunk.write(element_str)
            current_size = element_len
        else:
            current_chunk.write(element_str)
            current_size += element_len
    
    if current_size > 0:
        chunks.append(current_chunk.getvalue())
    
    return chunks

def count_keyword_occurrences(content: str, keywords: list) -> dict:
    """
    Count occurrences of each keyword in the content.
    Returns a dictionary mapping keywords to occurrence counts.
    """
    if not content or not keywords:
        return {}
    
    # Create soup to extract text content only (to ignore HTML tags)
    if BeautifulSoup:
        try:
            soup = BeautifulSoup(content, "html.parser")
            text_content = soup.get_text()
        except:
            text_content = content
    else:
        text_content = re.sub(r'<[^>]+>', '', content)  # Basic HTML tag removal fallback
    
    counts = {}
    for keyword in keywords:
        if keyword and keyword.strip():
            counts[keyword] = text_content.count(keyword)
    
    return counts

def optimize_content(content: str, primary_keyword: str = "", secondary_keywords: list = None, internal_links: list = None, site_name: str = None) -> dict:
    """
    Optimize Thai HTML content for keywords. If content is too long, split it into chunks,
    optimize each using fallback logic, and combine results.
    Ensures adequate primary keyword density throughout the content.
    Compares keyword occurrences before and after optimization.
    
    Args:
        content: The HTML content to optimize
        primary_keyword: The primary keyword to optimize for
        secondary_keywords: List of secondary keywords
        site_name: Site name for internal link lookup (optional)
    """
    secondary_keywords = secondary_keywords or []
    content_hash = hashlib.md5(content.encode()).hexdigest()
    keywords_hash = hashlib.md5(f"{primary_keyword}{''.join(secondary_keywords)}".encode()).hexdigest()
    site_hash = hashlib.md5(f"{site_name or ''}".encode()).hexdigest()[:8]
    cache_key = f"optimization_cache_{content_hash}_{keywords_hash}_{site_hash}"
    cached_result = st.session_state.get(cache_key)
    if cached_result:
        st.success("Retrieved from cache!")
        return cached_result
        
    # Use manually entered internal links if provided
    if internal_links:
        logging.info(f"[InternalLinks] Using {len(internal_links)} manually entered internal links")
    else:
        internal_links = []
        logging.info("[InternalLinks] No internal links provided")
    
    total_length = len(content)
    if total_length < 10000:
        CHUNK_LIMIT = 6000  # Reduced from 8000
    elif total_length < 50000:
        CHUNK_LIMIT = 12000  # Reduced from 20000
    else:
        CHUNK_LIMIT = 10000  # Significantly reduced from 30000
    
    if total_length <= CHUNK_LIMIT:
        result = optimize_chunk_with_fallback(content, primary_keyword, secondary_keywords, internal_links=internal_links, site_name=site_name)
        if result:
            st.session_state[cache_key] = result
        return result
    else:
        chunks = smart_chunk_html(content, CHUNK_LIMIT)
        actual_chunks = len(chunks)
        progress_bar = st.progress(0)
        status_text = st.empty()
        status_text.text(f"Optimizing {actual_chunks} chunks...")
        optimized_chunks = []
        seo_elements = {
            'titles': [],
            'meta_descriptions': [],
            'alt_text': "",
            'wordpress_slug': ""
        }
        primary_keyword_inserted = False
        failed_chunks = []
        for i, chunk in enumerate(chunks):
            progress = i / actual_chunks
            progress_bar.progress(progress)
            status_text.text(f"Optimizing chunk {i+1}/{actual_chunks}...")
            
            # For the first chunk, include primary keyword. For others, just secondary keywords
            chunk_primary = ""
            if primary_keyword and not primary_keyword_inserted and i == 0:
                chunk_primary = primary_keyword
                primary_keyword_inserted = True
                
            # Only pass internal links to the first chunk to avoid duplicate links
            chunk_internal_links = internal_links if i == 0 and internal_links else []
            
            result = optimize_chunk_with_fallback(chunk, chunk_primary, secondary_keywords, site_name=site_name, internal_links=chunk_internal_links)
            if result is None:
                failed_chunks.append(i)
                continue
            optimized_chunks.append(result['optimized_html'])
            if result['titles'] and any(title.strip() for title in result['titles']):
                seo_elements['titles'].extend(result['titles'])
            if result['meta_descriptions'] and any(desc.strip() for desc in result['meta_descriptions']):
                seo_elements['meta_descriptions'].extend(result['meta_descriptions'])
            if result['alt_text'] and result['alt_text'].strip():
                if not seo_elements['alt_text']:
                    seo_elements['alt_text'] = result['alt_text']
            if result.get('wordpress_slug', '') and result.get('wordpress_slug', '').strip():
                if not seo_elements['wordpress_slug']:
                    seo_elements['wordpress_slug'] = result.get('wordpress_slug', '')
            time.sleep(1)
        progress_bar.progress(1.0)
        # Add retry logic for failed chunks with simplified parameters
        if failed_chunks:
            status_text.text(f"Attempting to retry {len(failed_chunks)} failed chunks with simplified parameters...")
            retried_chunks = []
            for chunk_idx in failed_chunks:
                # Try again with no keywords and simplified mode for problematic chunks
                retry_result = _optimize_single(chunks[chunk_idx], "", [], max_retries=2, simplified_mode=True)
                if retry_result is not None:
                    # Insert at the correct position to maintain order
                    optimized_chunks.insert(chunk_idx, retry_result['optimized_html'])
                    retried_chunks.append(chunk_idx)
                    # Update progress
                    progress = (i + 1 + len(retried_chunks)) / actual_chunks
                    progress_bar.progress(progress)
                    status_text.text(f"Recovered chunk {chunk_idx+1}")
                    time.sleep(1)
            
            # Update failed chunks list
            remaining_failed = [idx for idx in failed_chunks if idx not in retried_chunks]
            
            # Final desperate attempt for any remaining failed chunks - split them into tiny chunks
            if remaining_failed:
                status_text.text(f"Final attempt for {len(remaining_failed)} still-failing chunks using micro-chunking...")
                still_failed = []
                for chunk_idx in remaining_failed:
                    # Break into much smaller pieces
                    micro_chunk_size = min(3000, len(chunks[chunk_idx]) // 2)
                    micro_chunks = []
                    for i in range(0, len(chunks[chunk_idx]), micro_chunk_size):
                        micro_chunks.append(chunks[chunk_idx][i:i+micro_chunk_size])
                    
                    micro_optimized = []
                    all_micro_successful = True
                    for micro_idx, micro_chunk in enumerate(micro_chunks):
                        micro_result = _optimize_single(micro_chunk, "", [], max_retries=1, simplified_mode=True)
                        if micro_result is None:
                            all_micro_successful = False
                            break
                        micro_optimized.append(micro_result['optimized_html'])
                    
                    if all_micro_successful:
                        # All micro-chunks optimized successfully
                        optimized_chunks.insert(chunk_idx, "".join(micro_optimized))
                        status_text.text(f"Recovered chunk {chunk_idx+1} using micro-chunking")
                    else:
                        still_failed.append(chunk_idx)
                
                # Final update on any chunks that still failed
                if still_failed:
                    status_text.text(f"Warning: {len(still_failed)}/{actual_chunks} chunks failed all optimization attempts.")
                    st.error(f"Failed chunk indices: {still_failed}. Optimization may be incomplete.")
                    if not optimized_chunks:
                        return None
                else:
                    status_text.text("All chunks successfully optimized after multiple retry strategies!")
            else:
                status_text.text("All chunks successfully optimized after retry!")
        else:
            status_text.text("Optimization complete!")
        combined_html = "".join(optimized_chunks)
        final_result = {
            'optimized_html': combined_html,
            'titles': seo_elements['titles'],
            'meta_descriptions': seo_elements['meta_descriptions'],
            'alt_text': seo_elements['alt_text'],
            'wordpress_slug': seo_elements['wordpress_slug']
        }
        st.session_state[cache_key] = final_result
        return final_result

# Make sure the link database directory exists
os.makedirs(DB_DIR, exist_ok=True)

# Initialize optimization history cache
if 'history' not in st.session_state:
    st.session_state['history'] = []

# Load WordPress credentials if available
def init_wordpress_credentials():
    # For each site in WP_API_ENDPOINTS, get credentials from environment variables
    for site, config in WP_API_ENDPOINTS.items():
        user_env = f"WP_{site}_USER"
        pass_env = f"WP_{site}_PASSWORD"
        
        username = os.getenv(user_env)
        app_password = os.getenv(pass_env)
        
        if username and app_password:
            WP_API_ENDPOINTS[site]["username"] = username
            WP_API_ENDPOINTS[site]["app_password"] = app_password
            logging.info(f"Loaded WordPress credentials for {site}")
        else:
            logging.warning(f"WordPress credentials not found for {site}. Set {user_env} and {pass_env} environment variables.")
    
    return WP_API_ENDPOINTS

# Initialize OpenRouter client
try:
    openrouter_config = init_openrouter_client()
    st.sidebar.success(f"Using model: {openrouter_config['default_model']}")
except Exception as e:
    st.error(f"Failed to initialize OpenRouter client: {str(e)}")
    st.stop()

# Streamlit UI
st.title("Thai Content Keyword Optimizer")
st.subheader("SEO Optimization for Thai HTML Content")

upload_method = st.radio("Choose input method:", ["Text Input", "File Upload"])
html_input = ""
if upload_method == "Text Input":
    html_input = st.text_area("Enter Thai HTML content to optimize:", height=200)
else:
    uploaded_file = st.file_uploader("Upload Thai HTML file", type=['html', 'htm', 'txt'])
    if uploaded_file:
        html_input = uploaded_file.getvalue().decode('utf-8')
        st.success(f"Loaded file: {uploaded_file.name} ({len(html_input)} characters)")
        with st.expander("Preview uploaded content"):
            st.code(html_input[:500] + "..." if len(html_input) > 500 else html_input, language="html")

# Multi-keyword input (one per line)
keyword_input = st.text_area(
    "Keywords (one per line, first keyword is primary):",
    placeholder="Primary keyword\nSecondary keyword 1\nSecondary keyword 2",
    height=100
)

# Parse keywords
keywords = [k.strip() for k in keyword_input.split('\n') if k.strip()]
primary_keyword = keywords[0] if keywords else ""
secondary_keywords = keywords[1:] if len(keywords) > 1 else []

# Display parsed keywords for clarity
if keywords:
    st.info(f"Primary keyword: {primary_keyword}")
    if secondary_keywords:
        st.info(f"Secondary keywords: {', '.join(secondary_keywords)}")

# Manual internal link input
with st.expander("Add/Update Internal Links"):
    st.info("Paste a list of internal links below (Title on one line, URL on the next). The AI will intelligently insert them into the article above.")
    
    internal_links_input = st.text_area(
        "Potential Internal Links (Title/URL pairs):",
        height=200,
        help="Enter each link as a pair: title on one line, full URL on the next line"
    )
    
    # Process the internal links input into a usable format
    manual_internal_links = []
    
    if internal_links_input:
        lines = internal_links_input.strip().split('\n')
        for i in range(0, len(lines) - 1, 2):
            if i + 1 < len(lines):  # Make sure we have a pair
                title = lines[i].strip()
                url = lines[i + 1].strip()
                if title and url:  # Both title and URL must be non-empty
                    manual_internal_links.append({
                        "title": title,
                        "url": url,
                        "keywords": []  # No keywords associated with manual links
                    })

# WordPress submission functionality
def submit_to_wordpress(site_name, title, content, excerpt="", status="draft", categories=None, tags=None):
    """
    Submit a post to WordPress via REST API
    
    Args:
        site_name: Name of the site (must match a key in WP_API_ENDPOINTS)
        title: Post title
        content: Post content (HTML)
        excerpt: Post excerpt/summary
        status: Post status (draft, publish, etc)
        categories: List of category IDs
        tags: List of tag IDs
    
    Returns:
        Response from WordPress API or None if submission failed
    """
    site_config = WP_API_ENDPOINTS.get(site_name.upper())
    if not site_config:
        logging.error(f"No WordPress configuration found for site: {site_name}")
        return None
    
    if not site_config.get("username") or not site_config.get("app_password"):
        logging.error(f"WordPress credentials missing for {site_name}")
        return None
    
    # Prepare post data
    post_data = {
        "title": title,
        "content": content,
        "status": status
    }
    
    if excerpt:
        post_data["excerpt"] = excerpt
        
    if categories:
        post_data["categories"] = categories
        
    if tags:
        post_data["tags"] = tags
    
    # Set up authentication
    auth = (site_config["username"], site_config["app_password"])
    
    try:
        # Send POST request to WordPress API
        response = requests.post(
            site_config["api_url"],
            json=post_data,
            auth=auth,
            headers={"Content-Type": "application/json"}
        )
        
        # Check response
        if response.status_code in [200, 201]:
            logging.info(f"Successfully submitted post to {site_name}")
            return response.json()
        else:
            logging.error(f"WordPress API error: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        logging.error(f"Error submitting to WordPress: {str(e)}")
        return None

with st.expander("Advanced Options"):
    st.info("These settings help optimize processing of very large documents.")
    save_intermediate = st.checkbox("Save intermediate results (recommended for large files)", value=True)
    clear_cache = st.button("Clear cache")
    
    if clear_cache:
        cache_keys = [k for k in st.session_state.keys() if k.startswith('optimization_cache_')]
        for key in cache_keys:
            del st.session_state[key]
        st.success("Cache cleared!")

if st.button("Optimize Content"):
    if not html_input:
        st.error("Please enter Thai HTML content to optimize")
    else:
        line_count = html_input.count('\n') + 1
        char_count = len(html_input)
        tag_count = len(re.findall(r'<[^>]+>', html_input))
        
        st.info(f"Document stats: {line_count} lines, {char_count} characters, approximately {tag_count} HTML tags")
        
        start_time = time.time()
        with st.spinner("Analyzing document..."):
            try:
                result = optimize_content(html_input, primary_keyword, secondary_keywords, internal_links=manual_internal_links)
            except Exception as e:
                st.error(f"Optimization failed: {str(e)}")
                result = None

            end_time = time.time()
            process_time = end_time - start_time

            if result:
                st.success(f"Optimization complete in {process_time:.1f} seconds!")
                # Save the result to history
                history_entry = {
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'input_length': len(html_input),
                    'output_length': len(result['optimized_html']),
                    'keyword': primary_keyword if primary_keyword else "None",
                    'site_name': site_name if site_name else "None",
                    'process_time': process_time
                }
                

                st.session_state['history'].append(history_entry)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        label="Download Optimized HTML",
                        data=result['optimized_html'],
                        file_name="optimized.html",
                        mime="text/html"
                    )
                with col2:
                    json_data = json.dumps(result, ensure_ascii=False, indent=2)
                    st.download_button(
                        label="Download JSON (All Results)",
                        data=json_data,
                        file_name="optimization_results.json",
                        mime="application/json"
                    )
                
                with st.expander("Optimized HTML Content", expanded=True):
                    st.code(result['optimized_html'], language='html')
                
                with st.expander("SEO Elements", expanded=True):
                    st.write("Titles:", result['titles'])
                    st.write("Meta Descriptions:", result['meta_descriptions'])
                    st.write("Alt Text:", result['alt_text'])
                    st.write("WordPress Slug:", result.get('wordpress_slug', ''))
                    
                    # Check if internal links were successfully inserted
                    if manual_internal_links:
                        # Check each URL from our manual links
                        inserted_links = []
                        for link in manual_internal_links:
                            url = link['url']
                            if url in result['optimized_html']:
                                inserted_links.append(url)
                        
                        if inserted_links:
                            st.success(f"✅ Successfully inserted {len(inserted_links)} internal links")
                            with st.expander("Internal Links Details"):
                                for i, url in enumerate(inserted_links):
                                    st.write(f"{i+1}. {url}")
                        else:
                            st.warning("⚠️ No internal links were inserted. The content may not have had suitable placement opportunities.")
                
                # Keyword usage analysis
                if primary_keyword:
                    if BeautifulSoup:
                        try:
                            soup = BeautifulSoup(html_input, "html.parser")
                            original_text = soup.get_text()
                        except:
                            original_text = html_input
                    else:
                        original_text = re.sub(r'<[^>]+>', '', html_input)  # Basic HTML tag removal fallback
                    
                    primary_before = original_text.count(primary_keyword)
                    primary_count = result['optimized_html'].count(primary_keyword)
                    primary_status = '✅' if primary_count >= 3 else '⚠️'
                    st.write(f"{primary_status} Primary keyword '{primary_keyword}' appears {primary_count} times in the optimized content (was {primary_before} before).")
                    st.write("")

                # Display secondary keyword stats
                for keyword in secondary_keywords:
                    if keyword:
                        keyword_before = original_text.count(keyword) if 'original_text' in locals() else 0
                        keyword_count = result['optimized_html'].count(keyword)
                        keyword_status = '✅' if keyword_count >= 2 else '⚠️'
                        st.write(f"{keyword_status} Secondary keyword '{keyword}' appears {keyword_count} times in the optimized content (was {keyword_before} before).")
                st.write("")

                # Summary of keyword coverage
                if secondary_keywords:
                    missing_keywords = [keyword for keyword in secondary_keywords if result['optimized_html'].count(keyword) == 0]
                    coverage = ((len(secondary_keywords) - len(missing_keywords)) / len(secondary_keywords)) * 100
                    if coverage == 100:
                        st.success(f"🎯 All keywords successfully integrated in the optimized content!")
                    else:
                        st.warning(f"⚠️ {coverage:.1f}% keyword coverage achieved. Missing keywords: {', '.join(missing_keywords)}.")

                # WordPress submission section
                with st.expander("Submit to WordPress", expanded=False):
                    st.write("Submit this optimized content to WordPress:")
                    title_input = st.text_input("Post Title", 
                                             value=result.get('titles', [''])[0] if result.get('titles') else "")
                    
                    # Allow the user to select a site for submission
                    site_options = list(WP_API_ENDPOINTS.keys())
                    selected_site = st.selectbox("Select WordPress site", options=site_options)
                    
                    status_options = ["draft", "publish", "pending"]
                    status = st.selectbox("Post Status", options=status_options, index=0)
                    
                    meta_desc = result.get('meta_descriptions', [''])[0] if result.get('meta_descriptions') else ""
                    excerpt = st.text_area("Excerpt/Summary", value=meta_desc, height=100)
                    
                    if st.button("Submit to WordPress"):
                        if selected_site:
                            # First initialize credentials
                            init_wordpress_credentials()
                            
                            site_config = WP_API_ENDPOINTS.get(selected_site)
                            if not site_config:
                                st.error(f"No API endpoint configured for {selected_site}")
                            elif not site_config.get("username") or not site_config.get("app_password"):
                                st.error(f"WordPress credentials missing for {selected_site}. Please set environment variables.")
                            else:
                                with st.spinner("Submitting to WordPress..."):
                                    wp_response = submit_to_wordpress(
                                        site_name=selected_site,
                                        title=title_input,
                                        content=result['optimized_html'],
                                        excerpt=excerpt,
                                        status=status
                                    )
                                    
                                    if wp_response:
                                        post_url = wp_response.get('link', '')
                                        st.success(f"Post created successfully! URL: {post_url}")
                                        
                                        # Display post details
                                        post_details = {
                                            "title": wp_response['title']['rendered'] if 'title' in wp_response else title_input,
                                            "id": wp_response.get('id', ''),
                                            "status": wp_response.get('status', status),
                                            "url": post_url
                                        }
                                        st.json(post_details)
                                        
                                        # Update internal link database
                                        # Check if the selected site corresponds to a site in SITE_BASE_PATHS
                                        site_name_for_links = None
                                        for site_key in SITE_BASE_PATHS.keys():
                                            if site_key.upper() == selected_site.upper():
                                                site_name_for_links = site_key
                                                break
                                        
                                        if site_name_for_links and primary_keyword:
                                            try:
                                                # Extract slug from response or use the one from optimization
                                                seo_slug = result.get('wordpress_slug', "")
                                                if not seo_slug and post_url:
                                                    slug_match = re.search(r'/([^/]+)/?$', post_url)
                                                    if slug_match:
                                                        seo_slug = slug_match.group(1)
                                                
                                                update_link_database(
                                                    site_name=site_name_for_links,
                                                    article_main_title=title_input,
                                                    article_meta_desc=excerpt,
                                                    seo_slug=seo_slug,
                                                    primary_keyword=primary_keyword,
                                                    wp_post_url=post_url
                                                )
                                                st.success(f"🔗 Updated internal link database for {site_name_for_links} with keyword '{primary_keyword}'")
                                                
                                                # Show current links in database for this keyword
                                                relevant_links = get_relevant_internal_links(primary_keyword, site_name_for_links)
                                                if relevant_links:
                                                    with st.expander(f"Current internal links for keyword '{primary_keyword}'"):
                                                        for i, link in enumerate(relevant_links):
                                                            st.write(f"{i+1}. {link['title']} - {link['url']}")
                                            except Exception as e:
                                                st.warning(f"Failed to update internal link database: {str(e)}")
                                    else:
                                        st.error("Failed to submit to WordPress. Check logs for details.")

with st.sidebar.expander("Optimization History"):
    if not st.session_state['history']:
        st.write("No optimizations yet")
    else:
        for i, entry in enumerate(st.session_state['history']):
            st.write(f"**{entry['timestamp']}**")
            st.write(f"Size: {entry['input_length']} → {entry['output_length']} chars")
            st.write(f"Keyword: {entry['keyword']}")
            st.write(f"Time: {entry['process_time']:.1f} seconds")
            st.write("---")
