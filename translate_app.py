import streamlit as st
import streamlit.components.v1 as components
import os
import json
import html
import re
import time
import requests
import hashlib
import base64
from io import StringIO
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- MUST BE FIRST STREAMLIT COMMAND ---
st.set_page_config(
    page_title="Crypto HTML Translator",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize ALL session state keys
if 'history' not in st.session_state: st.session_state['history'] = []
if 'debug_log' not in st.session_state: st.session_state['debug_log'] = ""
if 'html_input' not in st.session_state: st.session_state['html_input'] = ""
if 'translation_result' not in st.session_state: st.session_state['translation_result'] = None
if 'publish_result' not in st.session_state: st.session_state['publish_result'] = None
if 'publish_error' not in st.session_state: st.session_state['publish_error'] = None
if 'wp_post_data' not in st.session_state: st.session_state['wp_post_data'] = None
if 'workflow_step' not in st.session_state: st.session_state['workflow_step'] = 'idle'
if 'auto_keywords' not in st.session_state: st.session_state['auto_keywords'] = ""
if 'wp_slug' not in st.session_state: st.session_state['wp_slug'] = ""
if 'last_success' not in st.session_state: st.session_state['last_success'] = None
if 'last_error' not in st.session_state: st.session_state['last_error'] = None

# Try to load environment variables from a .env file if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # Non-fatal: we'll rely on OS environment; provide an info message once
    st.info("python-dotenv not installed — reading environment only from OS. To use a .env file, install python-dotenv.")

# Try to import BeautifulSoup
try:
    from bs4 import BeautifulSoup
except ImportError:
    st.warning("BeautifulSoup is not installed. Attempting to install beautifulsoup4...")
    try:
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "beautifulsoup4"])
        from bs4 import BeautifulSoup
    except Exception:
        st.error("Failed to install BeautifulSoup. Falling back to basic chunking.")
        BeautifulSoup = None

def create_slug(text: str) -> str:
    """Create a URL-friendly slug from text."""
    if not text:
        return "default-slug"
    slug = re.sub(r'[^\w\s-]', '', text.lower())
    slug = re.sub(r'[\s-]+', '-', slug)
    return slug.strip('-')

# ----------------------
# WordPress Configuration & Auth
# ----------------------

def get_wp_config():
    """Retrieve WordPress configuration from environment variables."""
    return {
        'url': os.getenv('CRYPTODNES_WP_URL', 'https://cryptodnes.bg/th/').rstrip('/'),
        'username': os.getenv('CRYPTODNES_WP_USERNAME'),
        'app_password': os.getenv('CRYPTODNES_WP_APP_PASSWORD')
    }

def get_wp_auth_header():
    """Generate Basic Auth header for WordPress using env credentials."""
    config = get_wp_config()
    if not config['username'] or not config['app_password']:
        return {}
    credentials = f"{config['username']}:{config['app_password']}"
    token = base64.b64encode(credentials.encode()).decode()
    return {"Authorization": f"Basic {token}"}

def fetch_wp_post_by_url(url, post_id=None):
    """
    Fetch WordPress post data by URL or Post ID using the REST API.
    Returns the post data dict or raises an exception.
    """
    auth_header = get_wp_auth_header()
    
    parsed_url = urlparse(url)
    domain = f"{parsed_url.scheme}://{parsed_url.netloc}"
    
    path_parts = parsed_url.path.strip('/').split('/')
    subdir = ""
    if path_parts and len(path_parts[0]) == 2:
        subdir = f"/{path_parts[0]}"

    if post_id:
        try:
            for api_base in [f"{domain}{subdir}", domain]:
                api_root = f"{api_base}/wp-json"
                for ep in ["posts", "pages"]:
                    api_url = f"{api_root}/wp/v2/{ep}/{post_id}?_embed"
                    try:
                        response = requests.get(api_url, headers=auth_header, timeout=15)
                        if response.status_code == 200:
                            data = response.json()
                            if isinstance(data, dict) and 'title' in data:
                                return data
                    except:
                        continue
        except Exception as e:
            # We don't necessarily want to stop here as we can try URL discovery next
            pass

    try:
        # Step 1: Try to get the REST API link from the page headers
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        headers.update(auth_header)
        
        # We need to get the response to check headers AND potentially the body for the API link
        resp = requests.get(url, headers=headers, timeout=15, allow_redirects=True)
        if resp.status_code != 200:
            # Fallback to slug search later if direct load fails
            pass
        else:
            api_link = resp.headers.get('Link')
            
            post_api_url = None
            if api_link:
                match = re.search(r'<([^>]+)>;\s*rel="https://api.w.org/"', api_link)
                if not match:
                    match = re.search(r'<([^>]+)>;\s*rel="alternate";\s*type="application/json"', api_link)
                
                if match:
                    post_api_url = match.group(1)
            
            # If not in headers, check HTML for <link rel='https://api.w.org/' href='...' />
            if not post_api_url:
                soup = BeautifulSoup(resp.text, 'html.parser')
                link_tag = soup.find('link', rel='https://api.w.org/')
                if link_tag:
                    post_api_url = link_tag.get('href')

            # Step 2: If we found a direct API URL, use it
            if post_api_url:
                if '?' in post_api_url:
                    api_url = post_api_url + "&_embed"
                else:
                    api_url = post_api_url + "?_embed"
                
                response = requests.get(api_url, headers=headers, timeout=15)
                if response.status_code == 200:
                    data = response.json()
                    # Handle list response (e.g. if the link was to a collection)
                    if isinstance(data, list) and len(data) > 0:
                        data = data[0]
                    
                    if isinstance(data, dict) and 'title' in data:
                        return data

        # Step 3: Slug search fallback
        slug = path_parts[-1] if path_parts else ""
        if not slug:
            # Maybe the slug is the second to last part if URL ends with /
            if len(path_parts) > 1 and not path_parts[-1]:
                slug = path_parts[-2]
        
        if not slug:
            raise ValueError("Could not determine slug from URL")

        # Try these endpoints in order, with and without subdirectory
        api_roots = [f"{domain}{subdir}/wp-json", f"{domain}/wp-json"]
        endpoints = ["posts", "pages"]
        
        for api_root in api_roots:
            for ep in endpoints:
                search_url = f"{api_root}/wp/v2/{ep}?slug={slug}&_embed"
                try:
                    response = requests.get(search_url, headers=headers, timeout=15)
                    if response.status_code == 200:
                        data = response.json()
                        if isinstance(data, list) and len(data) > 0:
                            if 'title' in data[0]:
                                return data[0]
                except Exception:
                    continue

        raise ValueError(f"Could not find valid post/page data for '{slug}' or direct ID at {domain}")

    except Exception as e:
        raise Exception(f"Error fetching WordPress post: {str(e)}")

def publish_to_wp_th(content_data, categories=None, tags=None, featured_media_id=None, post_type="post", format="standard"):
    """
    Publish the translated content to the Thai WordPress site as a draft.
    Returns the response JSON or raises an exception.
    """
    config = get_wp_config()
    if not config['username'] or not config['app_password']:
        raise Exception("WordPress credentials missing in .env")

    endpoint = "posts" if post_type == "post" else "pages"
    api_url = f"{config['url']}/wp-json/wp/v2/{endpoint}"
    auth_header = get_wp_auth_header()
    
    post_payload = {
        "title": content_data.get('titles', [""])[0],
        "content": content_data.get('translated_html', ""),
        "excerpt": content_data.get('meta_descriptions', [""])[0] if content_data.get('meta_descriptions') else "",
        "slug": content_data.get('wordpress_slug', ""),
        "status": "draft",
        "format": format
    }
    
    if post_type == "post":
        if categories: post_payload["categories"] = categories
        if tags: post_payload["tags"] = tags
        
    if featured_media_id:
        post_payload["featured_media"] = featured_media_id

    response = requests.post(api_url, headers=auth_header, json=post_payload, timeout=20)
    if response.status_code not in [200, 201]:
        raise Exception(f"WP API Error: {response.text}")
    return response.json()

def get_or_create_th_term(term_name, taxonomy="category"):
    """
    Find or create a term (category or tag) on the Thai site.
    """
    config = get_wp_config()
    auth_header = get_wp_auth_header()
    if not auth_header:
        return None

    # Map taxonomy to endpoint
    endpoint = "categories" if taxonomy == "category" else "tags"
    api_url = f"{config['url']}/wp-json/wp/v2/{endpoint}"
    
    try:
        # Search for existing term
        search_response = requests.get(f"{api_url}?search={re.escape(term_name)}", headers=auth_header, timeout=10)
        search_response.raise_for_status()
        terms = search_response.json()
        
        for term in terms:
            if term['name'].lower() == term_name.lower():
                return term['id']
                
        # If not found, create it
        create_payload = {"name": term_name}
        create_response = requests.post(api_url, headers=auth_header, json=create_payload, timeout=10)
        create_response.raise_for_status()
        return create_response.json()['id']
        
    except Exception as e:
        # Silently fail for term sync, not critical enough to stop the workflow
        return None

def get_or_create_th_category(en_category_name):
    return get_or_create_th_term(en_category_name, "category")

def get_or_create_th_tag(en_tag_name):
    return get_or_create_th_term(en_tag_name, "post_tag")

def update_session_state_with_post_data(post_data):
    """Update Streamlit session state with fetched WordPress post data."""
    if not post_data:
        return

    # 1. Basic Info
    title_text = post_data.get('title', {}).get('rendered', 'Untitled')
    st.session_state['wp_post_data'] = post_data
    
    # Get content and remove TOC shortcodes to avoid duplication
    content_html = post_data.get('content', {}).get('rendered', '')
    # Remove common TOC shortcodes
    toc_patterns = [
        r'\[toc[^\]]*\]',
        r'\[table_of_contents[^\]]*\]',
        r'\[ez-toc[^\]]*\]',
        r'\[toc_page[^\]]*\]'
    ]
    for pattern in toc_patterns:
        content_html = re.sub(pattern, '', content_html, flags=re.IGNORECASE)
    
    st.session_state['html_input'] = content_html
    st.session_state['wp_slug'] = post_data.get('slug', "")

    # 2. SEO Metadata (Yoast)
    meta_desc = ""
    focus_kw = ""
    if 'yoast_head_json' in post_data:
        meta_desc = post_data['yoast_head_json'].get('description', "")
        focus_kw = post_data['yoast_head_json'].get('focuskw', "")
    
    if not focus_kw and 'meta' in post_data:
        focus_kw = post_data['meta'].get('_yoast_wpseo_focuskw', "")

    st.session_state['wp_meta_desc'] = meta_desc

    # 3. Tags & Keywords Automation
    en_tags = []
    if '_embedded' in post_data and 'wp:term' in post_data['_embedded']:
        for term_list in post_data['_embedded']['wp:term']:
            for term in term_list:
                if term.get('taxonomy') == 'post_tag':
                    en_tags.append(term['name'])
    
    kw_lines = []
    if focus_kw:
        kw_lines.append(focus_kw)
    
    for tag in en_tags:
        if tag not in kw_lines:
            kw_lines.append(tag)
    
    if kw_lines:
        kws = "\n".join(kw_lines)
        st.session_state['auto_keywords'] = kws
    
    return title_text


def upload_media_to_wp_th(image_url, title=""):
    """
    Download image from English site and upload to Thai site.
    """
    config = get_wp_config()
    auth_header = get_wp_auth_header()
    if not auth_header:
        return None

    api_url = f"{config['url']}/wp-json/wp/v2/media"
    
    try:
        # 1. Download image
        img_resp = requests.get(image_url, timeout=15)
        img_resp.raise_for_status()
        
        # 2. Get filename from URL
        filename = urlparse(image_url).path.split('/')[-1]
        if not filename:
            filename = "featured-image.jpg"
            
        # 3. Upload to Thai site
        upload_headers = auth_header.copy()
        upload_headers.update({
            "Content-Disposition": f"attachment; filename={filename}",
            "Content-Type": img_resp.headers.get('Content-Type', 'image/jpeg')
        })
        
        upload_resp = requests.post(api_url, headers=upload_headers, data=img_resp.content, timeout=30)
        upload_resp.raise_for_status()
        return upload_resp.json()['id']
        
    except Exception as e:
        st.warning(f"Could not sync featured image: {str(e)}")
        return None


def clean_html_output(html_content: str) -> str:
    """Remove editor artefacts such as emoji <img> tags, redundant <span> wrappers and excessive new-lines.
    Falls back gracefully if BeautifulSoup is unavailable."""
    if not html_content:
        return html_content

    try:
        cleaned = html_content
        if 'BeautifulSoup' in globals() and BeautifulSoup:
            soup = BeautifulSoup(cleaned, 'html.parser')

            # 1. Remove <img> with class that includes "emoji"
            for img in soup.find_all('img'):
                classes = img.get('class', []) or []
                if any('emoji' in c for c in classes):
                    img.decompose()

            # 2. Strip out span wrappers from copy-pasted editor output while keeping inner text
            for span in soup.find_all('span'):
                span.unwrap()

            cleaned = str(soup)
        # Regex fallbacks / extra cleaning
        cleaned = re.sub(r'<img[^>]*class="emoji"[^>]*>', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'</?span[^>]*>', '', cleaned, flags=re.IGNORECASE)
        # Collapse multiple new-lines
        cleaned = re.sub(r'\n+', '\n', cleaned)
        return cleaned
    except Exception:
        # In worst case, just return original
        return html_content

# ----------------------
# OpenRouter Configuration
# ----------------------

def get_openrouter_config():
    """Retrieve OpenRouter configuration from environment variables."""
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        raise ValueError(
            "OpenRouter API key not found. Set OPENROUTER_API_KEY in your environment or .env file. "
            "Example .env:\nOPENROUTER_API_KEY=your_key_here\nOPENROUTER_MODEL=z-ai/glm-4.7"
        )
    
    return {
        'api_key': api_key,
        'default_model': os.getenv('OPENROUTER_MODEL', "z-ai/glm-4.7"),
        'api_url': "https://openrouter.ai/api/v1/chat/completions"
    }

def extract_json_safely(resp_text):
    """
    Attempts to extract a JSON object from the response text.
    Returns the parsed JSON object or None if extraction fails.
    """
    # Method 1: Try direct JSON loading first
    try:
        # First, try to load the entire response as JSON
        parsed = json.loads(resp_text, strict=False)
        if isinstance(parsed, dict) and 'translated_html' in parsed:
            # Validate that the result is complete
            if _validate_translation_result(parsed):
                return parsed
    except json.JSONDecodeError:
        # Continue with extraction methods if direct loading fails
        pass
    
    # Method 2: Try to extract JSON manually by getting content between first { and last }
    try:
        # Find the first opening brace
        start_index = resp_text.find('{')
        if start_index == -1:
            raise ValueError("No opening brace found")

        # Find the matching closing brace, accounting for nested structures
        brace_level = 0
        end_index = -1
        in_string = False
        escape_char = False
        
        for i, char in enumerate(resp_text[start_index:]):
            current_index = start_index + i
            
            if char == '"' and not escape_char:
                in_string = not in_string
            
            if not in_string:
                if char == '{':
                    brace_level += 1
                elif char == '}':
                    brace_level -= 1
                    if brace_level == 0:
                        end_index = current_index
                        break # Found the matching closing brace
            
            # Handle escape characters
            escape_char = char == '\\' and not escape_char

        if end_index == -1:
            raise ValueError("Matching closing brace not found")

        # Extract the potential JSON string
        json_str = resp_text[start_index:end_index+1]

        # Clean control characters
        json_str = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]', '', json_str)
        
        # Fix common JSON issues with HTML attributes
        json_str = _fix_json_html_attributes(json_str)

        try:
            # Try parsing the extracted JSON
            parsed = json.loads(json_str, strict=False)
            if isinstance(parsed, dict) and 'translated_html' in parsed:
                # Validate that the result is complete
                if _validate_translation_result(parsed):
                    return parsed
                else:
                    return None
        except json.JSONDecodeError:
            # Don't continue to Method 3 if we had a JSON structure but it was malformed
            return None

    except Exception:
        pass
    
    # Method 3: Manual JSON construction from the response text (only if Method 2 didn't find JSON structure)
    try:
        # Look for the translated_html key specifically since that's what we need
        translated_html_match = re.search(r'"translated_html"\s*:\s*"(.*?)(?<!\\)"(?=\s*,|\s*})', resp_text, re.DOTALL)
        titles_match = re.search(r'"titles"\s*:\s*\[(.*?)\]', resp_text, re.DOTALL)
        meta_desc_match = re.search(r'"meta_descriptions"\s*:\s*\[(.*?)\]', resp_text, re.DOTALL)
        alt_text_match = re.search(r'"alt_text"\s*:\s*"(.*?)(?<!\\)"(?=\s*,|\s*})', resp_text, re.DOTALL)
        wp_slug_match = re.search(r'"wordpress_slug"\s*:\s*"(.*?)(?<!\\)"(?=\s*,|\s*})', resp_text, re.DOTALL)
        
        # If we have at least the translated_html key, we can create a valid response
        if translated_html_match:
            result = {}
            
            # Extract and validate translated_html
            html_content = translated_html_match.group(1)
            # Check if the HTML content seems complete (not cut off mid-tag)
            if not _is_html_content_complete(html_content):
                return None
                
            result['translated_html'] = html.unescape(html_content)
            
            # Extract titles, handling potential HTML and quotes
            if titles_match:
                title_items = []
                # Extract individual title items from array
                title_items_match = re.finditer(r'"(.*?)(?<!\\)"', titles_match.group(1))
                for m in title_items_match:
                    title_items.append(html.unescape(m.group(1)))
                result['titles'] = title_items
            else:
                result['titles'] = []
            
            # Extract meta descriptions
            if meta_desc_match:
                meta_items = []
                # Extract individual meta items from array
                meta_items_match = re.finditer(r'"(.*?)(?<!\\)"', meta_desc_match.group(1))
                for m in meta_items_match:
                    meta_items.append(html.unescape(m.group(1)))
                result['meta_descriptions'] = meta_items
            else:
                result['meta_descriptions'] = []
            
            # Extract alt text if present
            if alt_text_match:
                result['alt_text'] = html.unescape(alt_text_match.group(1))
            else:
                result['alt_text'] = ""
            
            # Extract WordPress slug if present
            if wp_slug_match:
                result['wordpress_slug'] = wp_slug_match.group(1)
            else:
                # Create a default slug if none is found
                if result.get('titles') and result['titles']:
                    result['wordpress_slug'] = create_slug(result['titles'][0])
                else:
                    result['wordpress_slug'] = "default-slug"
            
            # Final validation
            if _validate_translation_result(result):
                return result
            else:
                return None
                
    except Exception:
        pass
    
    # If all methods fail, return None
    return None

def _fix_json_html_attributes(json_str):
    """Fix common JSON issues with HTML attributes containing quotes."""
    # This is a simple fix for the most common case: style attributes with unescaped quotes
    # Replace unescaped quotes in style attributes
    json_str = re.sub(r'style="([^"]*)"([^"]*)"([^"]*)"', r'style="\1\\\"\2\\\"\3"', json_str)
    
    # Fix other common HTML attribute quote issues
    json_str = re.sub(r'(href|src|alt|title|class|id)="([^"]*)"([^"]*)"([^"]*)"', r'\1="\2\\\"\3\\\"\4"', json_str)
    
    return json_str

def _is_html_content_complete(html_content):
    """Check if HTML content appears to be complete (not truncated)."""
    # Check for incomplete tags at the end
    if html_content.endswith('<') or html_content.endswith('</'):
        return False
    
    # Check for incomplete table structures
    if '<table>' in html_content and '</table>' not in html_content:
        return False
    
    # Check for incomplete div structures
    open_divs = html_content.count('<div')
    close_divs = html_content.count('</div>')
    if open_divs > close_divs + 2:  # Allow some tolerance
        return False
    
    # Check if content ends abruptly (very short content might be truncated)
    if len(html_content.strip()) < 50:
        return False
    
    return True

def _validate_translation_result(result):
    """Validate that a translation result is complete and valid."""
    if not isinstance(result, dict):
        return False
    
    # Check required fields
    required_fields = ['translated_html', 'titles', 'meta_descriptions', 'alt_text']
    for field in required_fields:
        if field not in result:
            return False
    
    # Check that translated_html is not empty and appears complete
    html_content = result.get('translated_html', '')
    if not html_content or len(html_content.strip()) < 50:
        return False
    
    # Check if HTML content appears complete
    if not _is_html_content_complete(html_content):
        return False
    
    # Check that lists are actually lists
    if not isinstance(result.get('titles'), list):
        return False
    if not isinstance(result.get('meta_descriptions'), list):
        return False
    
    return True


def _prepend_keyword(text: str, keyword: str) -> str:
    """Ensure keyword appears at the beginning of the provided text."""
    if not keyword:
        return text
    if not text or not text.strip():
        return keyword
    if keyword.lower() in text.lower():
        return text
    return f"{keyword} {text}".strip()


def _inject_keyword_into_html(html_content: str, keyword: str) -> str:
    """Insert keyword into the first textual HTML container when missing."""
    if not keyword:
        return html_content
    try:
        pattern = re.compile(r'<(p|h[1-6]|span|li|td|th|div|section)[^>]*>', re.IGNORECASE)
        match = pattern.search(html_content)
        if match:
            insert_pos = match.end()
            return f"{html_content[:insert_pos]}{keyword} {html_content[insert_pos:]}"
    except re.error:
        pass
    return f"{keyword} {html_content}".strip()


def _enforce_primary_keyword_usage(result: dict, primary_keyword: str) -> dict:
    """Guarantee that the primary keyword appears across key SEO fields."""
    if not primary_keyword or not isinstance(result, dict):
        return result

    keyword_lower = primary_keyword.lower()

    html_content = result.get('translated_html', '') or ''
    if keyword_lower not in html_content.lower():
        result['translated_html'] = _inject_keyword_into_html(html_content, primary_keyword)

    titles = result.get('titles')
    if isinstance(titles, list):
        keyword_in_titles = any(isinstance(title, str) and keyword_lower in title.lower() for title in titles)
        if not keyword_in_titles:
            if titles:
                titles[0] = _prepend_keyword(titles[0], primary_keyword)
            else:
                titles.append(primary_keyword)
        result['titles'] = titles
    elif titles is None:
        result['titles'] = [primary_keyword]

    meta_descriptions = result.get('meta_descriptions')
    if isinstance(meta_descriptions, list) and meta_descriptions:
        updated_meta = []
        for desc in meta_descriptions:
            if isinstance(desc, str) and desc.strip():
                if keyword_lower in desc.lower():
                    updated_meta.append(desc)
                else:
                    updated_meta.append(_prepend_keyword(desc, primary_keyword))
            else:
                updated_meta.append(primary_keyword)
        result['meta_descriptions'] = updated_meta

    alt_text = result.get('alt_text', '')
    if isinstance(alt_text, str):
        if not alt_text.strip():
            result['alt_text'] = primary_keyword
        elif keyword_lower not in alt_text.lower():
            result['alt_text'] = _prepend_keyword(alt_text, primary_keyword)

    return result

def _translate_single(text: str, primary_keyword: str = "", secondary_keywords: list = None, max_retries=3, simplified_mode=False, extra_context="") -> dict:
    """Translate a single chunk of HTML content using Claude via OpenRouter with retries."""
    secondary_keywords = secondary_keywords or []
    
    # Common core requirements for both modes
    core_rules = """
1. HTML Integrity: Keep all HTML tags, attributes, and inline styles completely unchanged.
2. Technical Content: Preserve URLs, variables (e.g. {{var}}), and shortcodes (e.g. [toc]) exactly.
3. Entities & Brands: Keep brand names, coin names, and technical entities in English (e.g., "Bitcoin").
4. Technical Terms: Keep technical terms in English but explain them in easy-to-understand Thai where appropriate.
5. JSON Output: Return ONLY a valid JSON object with this exact structure:
{
  "translated_html": "Thai content",
  "titles": ["3 options"],
  "meta_descriptions": ["3 options"],
  "alt_text": "Thai alt text",
  "wordpress_slug": "english-slug"
}"""

    if simplified_mode:
        prompt = f"""You are a Thai translator specializing in HTML content. Translate this HTML chunk to Thai following these rules:
{core_rules}

5. Only translate visible text.
6. Return ONLY the JSON object, no extra text.

Content to translate:
{text}"""
    else:
        # Detailed SEO and Journalistic requirements
        prompt = f"""You are a Senior Thai Crypto journalist. Translate HTML content to Thai following these rules:
{core_rules}

5. Translate ALL Visible Texts: This includes headers, paragraphs, list items, table cells, and especially anchor text in links.
6. URL Localization: Change ALL URLs containing /en/ to /th/ (e.g., https://cryptodnes.bg/en/terms-of-use/ → https://cryptodnes.bg/th/terms-of-use/).
7. Specific Link Mapping: Also change these domains to /th variants: bestwallettoken.com, bitcoinhyper.com, pepenode.io, maxidogetoken.com, wallstreetpepe.com.
8. Thai Ontological Flow: Use direct, active phrasing. Ensure proper spacing between Thai and English words.
9. Keyword Integration:
   - Primary Keyword: {primary_keyword if primary_keyword else "None"}
   - Secondary Keywords: {', '.join(secondary_keywords) if secondary_keywords else "None"}
   - Ensure natural placement in titles, first paragraph, and headings.

10. SEO Best Practices:
   - Provide 3 Thai title options (50-60 characters).
   - Provide 3 meta descriptions (150-160 characters) including keywords.
   - For wordpress_slug: If an original slug is provided in the CONTEXT below, use it EXACTLY without any additions. Otherwise, create a concise English URL-friendly version (3-5 words).

10. Chunk Context: This is a chunk of a larger document. Do not add or remove wrapper tags. Do not try to close tags started in other chunks.

Content to translate:
{text}

Return ONLY the JSON object, no commentary."""
    
    # Add extra context if provided
    if extra_context:
        prompt += f"\n\nCONTEXT:\n{extra_context}"

    # Get config
    try:
        config = get_openrouter_config()
    except Exception:
        return None
    
    for attempt in range(max_retries):
        try:
            response = requests.post(
                url=config['api_url'],
                headers={
                    "Authorization": f"Bearer {config['api_key']}",
                    "Content-Type": "application/json"
                },
                data=json.dumps({
                    "model": config['default_model'],
                    "messages": [{"role": "user", "content": prompt}]
                }),
                timeout=240
            )
            
            # Check for rate limiting
            if response.status_code == 429:
                wait_time = int(response.headers.get('Retry-After', 60))
                time.sleep(wait_time)
                continue
            
            response.raise_for_status()
            resp_data = response.json()
            
            if not resp_data or 'choices' not in resp_data or not resp_data['choices']:
                if attempt == max_retries - 1:
                    return None
                time.sleep(2)
                continue
            
            raw_text = resp_data['choices'][0]['message']['content']
            parsed = extract_json_safely(raw_text)
            
            if parsed:
                # Add wordpress_slug if missing
                if 'wordpress_slug' not in parsed and parsed.get('titles') and parsed['titles']:
                    parsed['wordpress_slug'] = create_slug(parsed['titles'][0])
                elif not parsed.get('wordpress_slug'):
                    parsed['wordpress_slug'] = "translated-post"
                
                parsed['translated_html'] = html.unescape(parsed.get('translated_html', ''))
                return parsed
            
            # If parsing failed, retry
            time.sleep(2)
            
        except Exception:
            if attempt == max_retries - 1:
                return None
            time.sleep(2)
    
    return None

def is_wordpress_gutenberg_content(content: str) -> bool:
    """
    Detect if the content contains WordPress Gutenberg blocks.
    """
    return bool(re.search(r'<!-- wp:', content))

def find_gutenberg_block_boundaries(content: str) -> list:
    """
    Find the start and end positions of WordPress Gutenberg blocks.
    Returns a list of tuples (start_pos, end_pos) for each block.
    """
    block_boundaries = []
    
    # Find all block markers (opening and closing)
    opening_pattern = re.compile(r'<!-- wp:[^\s>]+ .*?-->')
    closing_pattern = re.compile(r'<!-- /wp:[^\s>]+ -->')
    
    # Find all opening markers
    opening_matches = list(opening_pattern.finditer(content))
    
    # Find all closing markers
    closing_matches = list(closing_pattern.finditer(content))
    
    # Match opening and closing blocks
    for opening_match in opening_matches:
        opening_pos = opening_match.start()
        opening_text = opening_match.group(0)
        
        # Extract block name from opening marker
        opening_name_match = re.search(r'<!-- wp:([^\s>]+)', opening_text)
        if not opening_name_match:
            continue
            
        opening_name = opening_name_match.group(1)
        
        # Find corresponding closing marker
        for closing_match in closing_matches:
            closing_pos = closing_match.end()
            closing_text = closing_match.group(0)
            
            # Extract block name from closing marker
            closing_name_match = re.search(r'<!-- /wp:([^\s>]+)', closing_text)
            if not closing_name_match:
                continue
                
            closing_name = closing_name_match.group(1)
            
            # Check if names match and closing comes after opening
            if opening_name == closing_name and closing_pos > opening_pos:
                # Check if this is the closest matching closing tag
                block_content = content[opening_pos:closing_pos]
                # Count opening and closing tags of the same type within this range
                opening_count = len(re.findall(f'<!-- wp:{opening_name}[\\s>]', block_content))
                closing_count = len(re.findall(f'<!-- /wp:{opening_name} -->', block_content))
                
                # If counts match, we found the correct closing tag
                if opening_count == closing_count:
                    block_boundaries.append((opening_pos, closing_pos))
                    break
    
    return block_boundaries

def smart_chunk_wordpress_content(content: str, max_length: int) -> list:
    """
    Specialized chunking for WordPress Gutenberg content that preserves block integrity.
    """
    # If content doesn't contain WordPress blocks, use regular chunking
    if not is_wordpress_gutenberg_content(content):
        if not BeautifulSoup:
            return [content[i:i+max_length] for i in range(0, len(content), max_length)]
        else:
            return smart_chunk_html(content, max_length)
    
    # Find all Gutenberg block boundaries
    block_boundaries = find_gutenberg_block_boundaries(content)
    
    if not block_boundaries:
        # Fallback to regular chunking if no Gutenberg blocks found
        if not BeautifulSoup:
            return [content[i:i+max_length] for i in range(0, len(content), max_length)]
        else:
            return smart_chunk_html(content, max_length)
    
    # Sort blocks by start position
    block_boundaries.sort(key=lambda x: x[0])
    
    chunks = []
    current_chunk = StringIO()
    current_size = 0
    last_end = 0
    
    # Process content by respecting block boundaries
    for start, end in block_boundaries:
        # Add content between last block and current block
        if start > last_end:
            between_content = content[last_end:start]
            between_len = len(between_content)
            
            # If adding between content would exceed max_length, start a new chunk
            if current_size + between_len > max_length and current_size > 0:
                chunks.append(current_chunk.getvalue())
                current_chunk = StringIO()
                current_size = 0
            
            current_chunk.write(between_content)
            current_size += between_len
        
        # Process the current block
        block_content = content[start:end]
        block_len = len(block_content)
        
        # If the block itself exceeds max_length, we need to handle it specially
        if block_len > max_length:
            # If we have content in the current chunk, add it first
            if current_size > 0:
                chunks.append(current_chunk.getvalue())
                current_chunk = StringIO()
                current_size = 0
            
            # For large blocks, we keep them intact but as separate chunks
            chunks.append(block_content)
        else:
            # If adding this block would exceed max_length, start a new chunk
            if current_size + block_len > max_length and current_size > 0:
                chunks.append(current_chunk.getvalue())
                current_chunk = StringIO()
                current_chunk.write(block_content)
                current_size = block_len
            else:
                current_chunk.write(block_content)
                current_size += block_len
        
        last_end = end
    
    # Add any remaining content after the last block
    if last_end < len(content):
        remaining = content[last_end:]
        remaining_len = len(remaining)
        
        if current_size + remaining_len > max_length and current_size > 0:
            chunks.append(current_chunk.getvalue())
            current_chunk = StringIO()
            current_chunk.write(remaining)
        else:
            current_chunk.write(remaining)
    
    # Add the final chunk if there's anything left
    if current_chunk.getvalue():
        chunks.append(current_chunk.getvalue())
    
    return chunks

def translate_chunk_with_fallback(chunk, primary_keyword="", secondary_keywords=None, max_level=3, extra_context=""):
    """
    Attempts to translate a chunk using _translate_single.
    If translation fails and max_level is not exceeded, the chunk is split intelligently.
    Returns a dictionary with combined translated_html and aggregated SEO elements or None if translation fails.
    """
    secondary_keywords = secondary_keywords or []
    
    # First attempt: try to translate the whole chunk
    result = _translate_single(chunk, primary_keyword, secondary_keywords, extra_context=extra_context)
    if result is not None:
        return result
    
    # If we've reached the maximum recursion level or the chunk is too small, give up
    if max_level <= 0 or len(chunk) < 1000:
        return None
    
    # Check if this is WordPress Gutenberg content
    is_wp_content = is_wordpress_gutenberg_content(chunk)
    
    # For WordPress content, try to split at block boundaries
    if is_wp_content:
        block_boundaries = find_gutenberg_block_boundaries(chunk)
        if block_boundaries:
            # If we have multiple blocks, try to split between them
            if len(block_boundaries) > 1:
                # Find a good splitting point (middle block boundary)
                mid_idx = len(block_boundaries) // 2
                split_point = block_boundaries[mid_idx][0]
                
                chunk1 = chunk[:split_point]
                chunk2 = chunk[split_point:]
            else:
                # Only one block, split before and after it
                start, end = block_boundaries[0]
                if start > 0:
                    # Content before the block
                    chunk1 = chunk[:start]
                    # Block and content after it
                    chunk2 = chunk[start:]
                elif end < len(chunk):
                    # Block
                    chunk1 = chunk[:end]
                    # Content after the block
                    chunk2 = chunk[end:]
                else:
                    # Just split in the middle as a last resort
                    mid = len(chunk) // 2
                    chunk1 = chunk[:mid]
                    chunk2 = chunk[mid:]
        else:
            # No block boundaries found, split in the middle
            mid = len(chunk) // 2
            chunk1 = chunk[:mid]
            chunk2 = chunk[mid:]
    else:
        # For regular HTML, split in the middle
        mid = len(chunk) // 2
        chunk1 = chunk[:mid]
        chunk2 = chunk[mid:]
    
    # Try to translate each half recursively
    result1 = translate_chunk_with_fallback(chunk1, primary_keyword, secondary_keywords, max_level-1, extra_context=extra_context)
    result2 = translate_chunk_with_fallback(chunk2, "", secondary_keywords, max_level-1)
    
    # If either half failed, the whole chunk fails
    if result1 is None or result2 is None:
        return None
    
    # Combine the results
    combined_html = result1['translated_html'] + result2['translated_html']
    titles = result1['titles'] + result2['titles']
    meta_descriptions = result1['meta_descriptions'] + result2['meta_descriptions']
    alt_text = result1['alt_text'] if result1['alt_text'].strip() else result2['alt_text']
    wordpress_slug = result1.get('wordpress_slug', '') if result1.get('wordpress_slug', '').strip() else result2.get('wordpress_slug', '')
    
    return {
        'translated_html': combined_html,
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
    # Check if content contains WordPress Gutenberg blocks
    if is_wordpress_gutenberg_content(html_content):
        return smart_chunk_wordpress_content(html_content, max_length)
    
    if not BeautifulSoup:
        return [html_content[i:i+max_length] for i in range(0, len(html_content), max_length)]
    
    soup = BeautifulSoup(html_content, "html.parser")
    container = soup.body if soup.body else soup

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

def translate_content(content: str, primary_keyword: str = "", secondary_keywords: list = None, max_workers: int = 1, source_post_data: dict = None) -> dict:
    """
    Translate HTML content. If content is too long, split it into chunks,
    translate each using fallback logic, and combine results.
    """
    secondary_keywords = secondary_keywords or []
    
    # Extract source metadata if available
    source_title = ""
    source_meta = ""
    source_slug = ""
    if source_post_data:
        source_title = source_post_data.get('title', {}).get('rendered', "")
        source_slug = source_post_data.get('slug', "")
        if 'yoast_head_json' in source_post_data:
            source_meta = source_post_data['yoast_head_json'].get('description', "")
    
    content_hash = hashlib.md5(content.encode()).hexdigest()
    keywords_hash = hashlib.md5(f"{primary_keyword}{''.join(secondary_keywords)}".encode()).hexdigest()
    cache_key = f"translation_cache_{content_hash}_{keywords_hash}"
    cached_result = st.session_state.get(cache_key)
    if cached_result:
        st.success("Retrieved from cache!")
        return cached_result
    
    # Detect if content is WordPress Gutenberg
    is_wp_content = is_wordpress_gutenberg_content(content)
    
    # Adjust chunk limits based on content type and length
    total_length = len(content)
    if is_wp_content:
        # Use larger chunks for WordPress content to keep blocks together
        if total_length < 10000:
            CHUNK_LIMIT = 5000  # Smaller chunks for WP content
        elif total_length < 50000:
            CHUNK_LIMIT = 8000  # Reduced from 12000
        else:
            CHUNK_LIMIT = 8000  # Reduced from 10000
    else:
        # Regular HTML content
        if total_length < 10000:
            CHUNK_LIMIT = 6000
        elif total_length < 50000:
            CHUNK_LIMIT = 12000
        else:
            CHUNK_LIMIT = 15000  # Increased to reduce number of chunks
    
    # Pass source metadata context for the first chunk
    extra_context = ""
    if source_title or source_meta or source_slug:
        extra_context = "\n\nORIGINAL SEO METADATA (for context):\n"
        if source_title: extra_context += f"Title: {source_title}\n"
        if source_meta: extra_context += f"Meta Description: {source_meta}\n"
        if source_slug: extra_context += f"Original Slug: {source_slug}\n"

    if total_length <= CHUNK_LIMIT:
        result = translate_chunk_with_fallback(content, primary_keyword, secondary_keywords, extra_context=extra_context)
        if result:
            # Sanitize HTML to remove emojis / unnecessary spans
            result['translated_html'] = clean_html_output(result['translated_html'])
            result = _enforce_primary_keyword_usage(result, primary_keyword)
            st.session_state[cache_key] = result
        return result
    else:
        # Use the appropriate chunking method based on content type
        if is_wp_content:
            chunks = smart_chunk_wordpress_content(content, CHUNK_LIMIT)
        else:
            chunks = smart_chunk_html(content, CHUNK_LIMIT)
        actual_chunks = len(chunks)
        progress_bar = st.progress(0)
        status_text = st.empty()
        status_text.text(f"Translating {actual_chunks} chunks...")
        translated_chunks = [None] * actual_chunks
        seo_elements = {
            'titles': [],
            'meta_descriptions': [],
            'alt_text': "",
            'wordpress_slug': ""
        }
        primary_keyword_inserted = False
        failed_chunks = []

        # Parallel translation of chunks
        completed = 0
        def submit_task(i, chunk):
            # Determine primary keyword for the first chunk only
            chunk_primary = ""
            if primary_keyword and not primary_keyword_inserted and i == 0:
                # Note: we don't modify primary_keyword_inserted here to avoid race; we just apply it to i==0
                chunk_primary = primary_keyword
            
            # Pass source metadata context for the first chunk
            chunk_context = ""
            if i == 0 and extra_context:
                chunk_context = extra_context
                
            return translate_chunk_with_fallback(chunk, chunk_primary, secondary_keywords, extra_context=chunk_context)

        with ThreadPoolExecutor(max_workers=max(1, int(max_workers))) as executor:
            future_map = {executor.submit(submit_task, i, chunk): i for i, chunk in enumerate(chunks)}
            for future in as_completed(future_map):
                i = future_map[future]
                try:
                    result = future.result()
                except Exception:
                    result = None
                
                if result is None:
                    failed_chunks.append(i)
                else:
                    translated_chunks[i] = result['translated_html']
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
                completed += 1
                progress_value = completed / actual_chunks if actual_chunks else 0.0
                progress_bar.progress(min(1.0, max(0.0, progress_value)))
                status_text.text(f"Translated {completed}/{actual_chunks} chunks...")

        if failed_chunks:
            st.warning(f"Note: {len(failed_chunks)} chunks failed initial translation and will be retried.")

        progress_bar.progress(1.0)
        # Add retry logic for failed chunks with simplified parameters
        if failed_chunks:
            status_text.text(f"Attempting to retry {len(failed_chunks)} failed chunks with simplified parameters...")
            retried_chunks = []
            for chunk_idx in failed_chunks:
                # Try again with no keywords and simplified mode for problematic chunks
                retry_result = _translate_single(chunks[chunk_idx], "", [], max_retries=2, simplified_mode=True)
                if retry_result is not None:
                    translated_chunks[chunk_idx] = retry_result['translated_html']
                    retried_chunks.append(chunk_idx)
                    # Update progress
                    progress = (completed + len(retried_chunks)) / actual_chunks if actual_chunks else 1.0
                    progress_bar.progress(min(1.0, max(0.0, progress)))
                    status_text.text(f"Recovered chunk {chunk_idx+1}")
            
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
                    
                    micro_translated = []
                    all_micro_successful = True
                    for micro_idx, micro_chunk in enumerate(micro_chunks):
                        micro_result = _translate_single(micro_chunk, "", [], max_retries=1, simplified_mode=True)
                        if micro_result is None:
                            all_micro_successful = False
                            break
                        micro_translated.append(micro_result['translated_html'])
                    
                    if all_micro_successful:
                        # All micro-chunks translated successfully
                        translated_chunks[chunk_idx] = "".join(micro_translated)
                        status_text.text(f"Recovered chunk {chunk_idx+1} using micro-chunking")
                    else:
                        still_failed.append(chunk_idx)
                
                # Final update on any chunks that still failed
                if still_failed:
                    status_text.text(f"Warning: {len(still_failed)}/{actual_chunks} chunks failed all translation attempts.")
                    st.error(f"Failed chunk indices: {still_failed}. Translation may be incomplete.")
                    if not translated_chunks:
                        return None
                else:
                    status_text.text("All chunks successfully translated after multiple retry strategies!")
            else:
                status_text.text("All chunks successfully translated after retry!")
        else:
            status_text.text("Translation complete!")
        # Ensure order and drop any Nones before join (failed ones would have been retried)
        combined_html = clean_html_output("".join([c for c in translated_chunks if isinstance(c, str)]))
        final_result = {
            'translated_html': combined_html,
            'titles': seo_elements['titles'],
            'meta_descriptions': seo_elements['meta_descriptions'],
            'alt_text': seo_elements['alt_text'],
            'wordpress_slug': seo_elements['wordpress_slug']
        }
        final_result = _enforce_primary_keyword_usage(final_result, primary_keyword)
        st.session_state[cache_key] = final_result
        return final_result

# --- 1. Notification Area (Survives reruns) ---
if st.session_state.get('last_success'):
    st.success(st.session_state['last_success'])
    # Don't clear immediately, let it stay for the current rerun
if st.session_state.get('last_error'):
    st.error(st.session_state['last_error'])

# Streamlit UI
st.title("Crypto HTML Translation")
st.subheader("Thai Crypto Journalist Translation Tool")

# Workflow Status Banner
step = st.session_state['workflow_step']
if step == 'idle':
    st.info("💡 **Status: Ready** - Start by fetching a WordPress post or entering HTML manually.")
elif step == 'fetched':
    st.success("✅ **Status: Content Fetched** - WordPress content is loaded. Adjust keywords and click Translate.")
elif step == 'translated':
    st.success("🎨 **Status: Translated** - Review the translation below. You can now publish to WordPress.")
elif step == 'published':
    st.success("🚀 **Status: Published** - The draft has been created on the Thai site.")

# --- 2. Input Selection Section ---
if 'input_method_selector' not in st.session_state:
    st.session_state['input_method_selector'] = "Text Input"

upload_method = st.radio(
    "Choose input method:", 
    ["Text Input", "File Upload", "WordPress Fetch"], 
    key="input_method_selector"
)
html_input = ""

if upload_method == "WordPress Fetch":
    # Show WordPress Automation inside here
    fetch_expanded = st.session_state['workflow_step'] in ['idle', 'fetched']
    with st.expander("WordPress Automation", expanded=fetch_expanded):
        col1, col2 = st.columns([3, 1])
        with col1:
            wp_url_input = st.text_input("Paste English Post/Page URL:", value="https://cryptodnes.bg/en/terms-of-use/", key="wp_url_input")
        with col2:
            wp_id_input = st.text_input("OR Post ID:", placeholder="187200", key="wp_id_input")
            
        if st.button("Fetch Content from WordPress"):
            if not wp_url_input and not wp_id_input:
                st.error("Please enter a URL or Post ID")
            else:
                with st.spinner("Fetching post data..."):
                    # Clear results but keep previous history
                    st.session_state['translation_result'] = None
                    st.session_state['publish_result'] = None
                    st.session_state['publish_error'] = None
                    st.session_state['last_success'] = None
                    st.session_state['last_error'] = None
                    
                    # Try to extract ID
                    extracted_id = wp_id_input
                    if not extracted_id and wp_url_input and 'post=' in wp_url_input:
                        try:
                            extracted_id = re.search(r'post=(\d+)', wp_url_input).group(1)
                        except: pass
                    
                    try:
                        post_data = fetch_wp_post_by_url(wp_url_input, post_id=extracted_id)
                        if post_data:
                            title_text = update_session_state_with_post_data(post_data)
                            st.session_state['workflow_step'] = 'fetched'
                            st.session_state['last_success'] = f"Fetched: {title_text}"
                            st.rerun()
                    except Exception as e:
                        st.session_state['last_error'] = str(e)
                        st.rerun()

    if st.session_state['wp_post_data']:
        html_input = st.session_state['html_input']
        title_text = st.session_state['wp_post_data'].get('title', {}).get('rendered', 'Untitled')
        st.info(f"Using content from fetched WordPress post: {title_text}")
    else:
        st.warning("No WordPress post fetched yet.")
elif upload_method == "Text Input":
    html_input = st.text_area("Enter HTML content to translate:", height=200, key="html_input")
else:
    uploaded_file = st.file_uploader("Upload HTML file", type=['html', 'htm', 'txt'], key="html_file_uploader")
    if uploaded_file:
        content = uploaded_file.getvalue().decode('utf-8')
        if content != st.session_state['html_input']:
            st.session_state['html_input'] = content
            st.session_state['translation_result'] = None
            st.session_state['workflow_step'] = 'idle'
    html_input = st.session_state['html_input']

# Multi-keyword input
keyword_input = st.text_area(
    "Keywords (one per line, first keyword is primary):",
    placeholder="Primary keyword\nSecondary keyword 1",
    height=100,
    key="auto_keywords"
)

# Parse keywords
keywords = [k.strip() for k in keyword_input.split('\n') if k.strip()]
primary_keyword = keywords[0] if keywords else ""
secondary_keywords = keywords[1:] if len(keywords) > 1 else []

with st.expander("Advanced Options"):
    st.info("These settings help optimize translation of very large documents.")
    save_intermediate = st.checkbox("Save intermediate results (recommended for large files)", value=True, key="save_intermediate_check")
    concurrency = st.slider("Parallel requests", min_value=1, max_value=8, value=3, key="concurrency_slider")
    if st.button("Clear Translation Cache", key="clear_cache_btn"):
        cache_keys = [k for k in st.session_state.keys() if k.startswith('translation_cache_')]
        for key in cache_keys: del st.session_state[key]
        st.success("Cache cleared!")

# --- 4. Translation Trigger ---
if st.button("Translate", type="primary"):
    if not html_input:
        st.error("Please enter HTML content to translate")
    else:
        # Clear previous notifications
        st.session_state['last_success'] = None
        st.session_state['last_error'] = None
        
        with st.spinner("Translating..."):
            try:
                wp_post_data = st.session_state.get('wp_post_data')
                result = translate_content(
                    html_input, 
                    primary_keyword, 
                    secondary_keywords, 
                    max_workers=concurrency,
                    source_post_data=wp_post_data
                )
                
                if result:
                    st.session_state['translation_result'] = result
                    st.session_state['translation_html_input'] = html_input
                    st.session_state['translation_primary_keyword'] = primary_keyword
                    st.session_state['translation_secondary_keywords'] = secondary_keywords
                    st.session_state['workflow_step'] = 'translated'
                    st.session_state['last_success'] = "Translation complete!"
                    st.session_state['publish_result'] = None
                    
                    # Record in history
                    history_entry = {
                        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                        'input_length': len(html_input),
                        'output_length': len(result['translated_html']),
                        'keyword': primary_keyword if primary_keyword else "None",
                    }
                    if 'history' not in st.session_state: st.session_state['history'] = []
                    st.session_state['history'].append(history_entry)
                    
                    st.rerun()
            except Exception as e:
                st.session_state['last_error'] = f"Translation failed: {str(e)}"
                st.rerun()

# --- 5. Persistent Results Display ---
if st.session_state['workflow_step'] in ['translated', 'published'] and st.session_state['translation_result']:
    result = st.session_state['translation_result']
    html_input = st.session_state['translation_html_input']
    
    # Render stats, ratios, and previews
    st.divider()
    st.subheader("Translation Results")
    
    # Detailed Stats
    def normalize_html(h): return re.sub(r'\s+', ' ', h).strip()
    norm_in = normalize_html(html_input)
    norm_out = normalize_html(result['translated_html'])
    
    in_lines, out_lines = html_input.count('\n')+1, result['translated_html'].count('\n')+1
    in_chars, out_chars = len(norm_in), len(norm_out)
    in_tags = len(re.findall(r'<[^>]+>', norm_in))
    out_tags = len(re.findall(r'<[^>]+>', norm_out))
    
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Lines", f"{out_lines}", f"{out_lines - in_lines} diff")
    col_b.metric("Characters", f"{out_chars}", f"{out_chars - in_chars} diff")
    col_c.metric("HTML Tags", f"{out_tags}", f"{out_tags - in_tags} diff")

    if in_tags > 0 and (out_tags / in_tags) < 0.85:
        st.warning("⚠️ The translated output has significantly fewer HTML tags than the original.")

    col1, col2 = st.columns(2)
    with col1: st.download_button("Download HTML", result['translated_html'], "translated.html", "text/html")
    with col2: st.download_button("Download JSON", json.dumps(result, ensure_ascii=False, indent=2), "result.json", "application/json")
    
    # Preview
    st.subheader("Side-by-Side Preview")
    c_orig, c_trans = st.columns(2)
    with c_orig:
        st.markdown("#### Original")
        components.html(html_input, height=500, scrolling=True)
    with c_trans:
        st.markdown("#### Translated")
        components.html(result['translated_html'], height=500, scrolling=True)

    # --- 6. WordPress Publishing (Persistent) ---
    if st.session_state['wp_post_data']:
        st.divider()
        st.subheader("WordPress Publishing")
        
        # Extracted categories/tags
        en_categories = []
        en_tags = []
        wp_data = st.session_state['wp_post_data']
        if '_embedded' in wp_data and 'wp:term' in wp_data['_embedded']:
            for term_list in wp_data['_embedded']['wp:term']:
                for term in term_list:
                    if term.get('taxonomy') == 'category': en_categories.append(term['name'])
                    elif term.get('taxonomy') == 'post_tag': en_tags.append(term['name'])
        
        st.info(f"Syncing categories: {', '.join(en_categories)} | Tags: {', '.join(en_tags)}")

        if st.button("Publish Draft to Cryptodnes.bg/th", type="secondary"):
            # Clear previous notifications
            st.session_state['last_success'] = None
            st.session_state['last_error'] = None
            
            with st.spinner("Publishing to WordPress..."):
                try:
                    th_cat_ids = [get_or_create_th_category(c) for c in en_categories if get_or_create_th_category(c)]
                    th_tag_ids = [get_or_create_th_tag(t) for t in en_tags if get_or_create_th_tag(t)]
                    
                    en_media_url = ""
                    if '_embedded' in wp_data and 'wp:featuredmedia' in wp_data['_embedded']:
                        en_media_url = wp_data['_embedded']['wp:featuredmedia'][0].get('source_url', "")
                    
                    th_media_id = upload_media_to_wp_th(en_media_url, title=result['titles'][0]) if en_media_url else None
                    
                    source_type = wp_data.get('type', 'post')
                    source_format = wp_data.get('format', 'standard')
                    
                    pub_result = publish_to_wp_th(
                        result, 
                        categories=th_cat_ids, 
                        tags=th_tag_ids, 
                        featured_media_id=th_media_id, 
                        post_type=source_type, 
                        format=source_format
                    )
                    
                    if pub_result:
                        st.session_state['publish_result'] = pub_result
                        st.session_state['workflow_step'] = 'published'
                        st.session_state['last_success'] = "✅ Successfully published as draft!"
                        st.balloons()
                    else:
                        st.session_state['last_error'] = "Failed to publish draft. Check logs."
                    st.rerun()
                except Exception as e:
                    st.session_state['last_error'] = f"Publish error: {str(e)}"
                    st.rerun()

        # Show Persistent Publish Success
        if st.session_state['publish_result']:
            pr = st.session_state['publish_result']
            st.success(f"Draft Created! ID: {pr.get('id')}")
            st.markdown(f"**Preview:** [{pr.get('link')}]({pr.get('link')})")
            config = get_wp_config()
            edit_url = f"{config['url']}/wp-admin/post.php?post={pr.get('id')}&action=edit"
            st.markdown(f"**📝 Edit in WordPress:** [{edit_url}]({edit_url})")

# History in Sidebar
with st.sidebar.expander("Translation History"):
    for entry in st.session_state['history'][::-1]:
        st.write(f"**{entry['timestamp']}** | {entry['keyword']}")
        st.write("---")
