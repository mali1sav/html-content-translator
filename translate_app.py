import streamlit as st
import os
import json
import html
import re
import time
import requests
import hashlib
from io import StringIO
from concurrent.futures import ThreadPoolExecutor, as_completed
 
# Try to load environment variables from a .env file if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # Non-fatal: we'll rely on OS environment; provide an info message once
    st.info("python-dotenv not installed — reading environment only from OS. To use a .env file, install python-dotenv.")

# ----------------------
# Utility: Clean HTML output
# ----------------------

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

def init_openrouter_client():
    """Initialize OpenRouter connection parameters"""
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        raise ValueError(
            "OpenRouter API key not found. Set OPENROUTER_API_KEY in your environment or .env file. "
            "Example .env:\nOPENROUTER_API_KEY=your_key_here\nOPENROUTER_MODEL=google/gemini-2.5-pro-preview"
        )
    
    # Get model from environment variable or use default
    model = os.getenv('OPENROUTER_MODEL', "google/gemini-2.5-pro-preview")
    
    return {
        'api_key': api_key,
        'default_model': model,
        'api_url': "https://openrouter.ai/api/v1/chat/completions"
    }

def extract_json_safely(resp_text):
    """
    Attempts to extract a JSON object from the response text.
    Returns the parsed JSON object or None if extraction fails.
    """
    # Debug log the response (first 500 chars to avoid excessive length)
    debug_text = resp_text[:500] + '...' if len(resp_text) > 500 else resp_text
    st.session_state['debug_log'] = debug_text
    
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
                    st.warning("Method 2: Extracted JSON failed validation")
                    return None
        except json.JSONDecodeError as e:
            st.warning(f"Method 2 JSON parse error: {e}. Preview: {json_str[:200]}...")
            # Don't continue to Method 3 if we had a JSON structure but it was malformed
            return None

    except Exception as e:
        st.warning(f"Method 2 error: {str(e)}")
    
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
                st.warning("Method 3: HTML content appears incomplete")
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
                    from re import sub
                    title = result['titles'][0]
                    slug = sub(r'[^\w\s-]', '', title.lower())
                    slug = sub(r'[\s-]+', '-', slug)
                    result['wordpress_slug'] = slug
                else:
                    result['wordpress_slug'] = "default-slug"
            
            # Final validation
            if _validate_translation_result(result):
                return result
            else:
                st.warning("Method 3: Constructed result failed validation")
                return None
                
    except Exception as e:
        st.warning(f"Method 3 error: {str(e)}")
    
    # If all methods fail, log the response for debugging and return None
    st.error(f"All extraction methods failed. First 300 chars of response: {resp_text[:300]}")
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

def _translate_single(text: str, primary_keyword: str = "", secondary_keywords: list = None, max_retries=3, simplified_mode=False) -> dict:
    """Translate a single chunk of HTML content using Claude via OpenRouter with retries."""
    secondary_keywords = secondary_keywords or []
    
    if simplified_mode:
        # Use a simpler prompt with fewer requirements for problematic chunks
        prompt = f"""You are a Thai translator specializing in HTML content. Translate this HTML chunk to Thai following these critical requirements:

1. Never modify HTML tags or attributes - only translate visible text
2. Keep all technical terms and brand names in English
3. Preserve all URLs, variables, and code exactly as they are
4. Return ONLY a valid JSON object with this exact structure:
{{
  "translated_html": "HTML content with Thai text",
  "titles": ["Any title found"],
  "meta_descriptions": ["Any meta description found"],
  "alt_text": "Any alt text found",
  "wordpress_slug": "english-version-of-title"
}}

Content to translate:
{text}"""
    else:
        # Use the full detailed prompt for normal translation
        prompt = f"""You are a Senior Thai Crypto journalist specializing in translating technical crypto content. Translate HTML content to Thai by following these strict translation requirements:

1. Preserve HTML Structure
   Keep all HTML tags, attributes, and inline styles completely unchanged. Do not modify the HTML in any way except to translate the visible text.

2. Retain Technical Content
   Do not modify any technical elements such as URLs, placeholders (e.g., [cur_year], {{{{variable}}}}), and shortcodes (e.g., [su_note], [toc]). They must remain exactly as they are — except for the specific links detailed in point #5 below. Under no circumstances should any code, variable, or shortcode be altered.

3. Maintain English Technical Terms, Entities, and Brands
   Do not translate entity names, or brands. For example, always use "Bitcoin" (not a translated equivalent) and keep all other technical terms and entities in English. This includes cryptocurrency names, blockchain terms, etc. Keep Technical terms in English but explain it in the way that is easier to understand via Thai. For example "Market intelligence - ฟีเจอร์วิเคราะห์ตลาดที่ช่วยให้ข้อมูลเชิงลึกสำหรับการตัดสินใจทางการเงิน"

4. Translate *ALL* Visible Texts.
   You *must* translate *all* visible text into Thai *without exception*, unless it falls under rule #3 (technical terms/brands). This includes *everything* inside h2, h3, paragraphs, list items, table cells, button text, *and especially text within `<span>` and anchor text in hyperlink tags, no matter where they appear*. Double-check to ensure *every* piece of translatable text is translated into Thai. *There should be NO English or another language text remaining in the translation except for the exceptions outlined in rule #3.*

5. Modify Only These Specific Links to the Thai Variant
   If you encounter *only* these specific links (with or without "/en" after the domain), change them to use "/th" instead:

   - `https://bestwallettoken.com` or `https://bestwallettoken.com/en` → `https://bestwallettoken.com/th`
   - `https://bitcoinhyper.com` or `https://bitcoinhyper.com/en` → `https://bitcoinhyper.com/th`
   - `https://pepenode.io` or `https://pepenode.io/en` → `https://pepenode.io/th`
   - `https://maxidogetoken.com/` or `https://maxidogetoken.com/en` → `https://maxidogetoken.com/th`
   - `https://wallstreetpepe.com/` or `https://wallstreetpepe.com/en` → `https://wallstreetpepe.com/th`

   Do not alter any other URLs or domains besides these five. All other links must remain exactly as they are in the original HTML.

6. Ensure Proper Spacing and Punctuation
   Translate the visible text content to Thai. Follow Thai ontological structure flow (entity then action/event and then outcome) with direct, active phrasing, and avoid using - to connect sentences (. Make sure no extra spaces or punctuation errors are introduced. Insert appropriate spacing where needed between Thai and any English words or technical terms. If there are English words in Thai sentences, ensure they are written in title case. For example: Exchange แบบ Non-Custodial ผู้ใช้จะได้รับ Private Keys ของตน. 

   Avoid word-for-word translation. You must ensure the sentence is correct and resonates with Thai Crypto readers. If needed, you can adjust the position of HTML tags to allow for active (direct) sentence structures rather than passive ones that may originate from the original language sentences.

7. Do not add or remove wrapper tags (like <!DOCTYPE html>, <html>, <head>, or <body>) unless they already exist in the snippet.

8. Keyword Integration:
   Primary Keyword: {primary_keyword if primary_keyword else "None provided"}
   {f"Ensure that the primary keyword '{primary_keyword}' appears naturally in Title (1x), First paragraph (1x), and Headings and remaining paragraphs where they fit naturally, Meta description (1x). Maintain original language (whether Thai or English or mix)." if primary_keyword else "No primary keyword provided."}
   {"Use the primary keyword exactly as provided — do not replace it with transliterations, synonym spellings, or alternative terms." if primary_keyword else ""}
   
   Secondary Keywords: {', '.join(secondary_keywords) if secondary_keywords else "None provided"}
   {f"IMPORTANT: You MUST include EACH secondary keyword at least once in the translated content. Place secondary keywords strategically in:" if secondary_keywords else "No secondary keywords provided."}
   {f"   - At least 2-3 H2 or H3 headings" if secondary_keywords else ""}
   {f"   - Within paragraphs where they fit naturally" if secondary_keywords else ""}
   {f"   - In the FAQ section questions and answers (if present)" if secondary_keywords else ""}
   
   {f"Priority secondary keywords (include these first):" if secondary_keywords else ""}
   {f"   {', '.join(secondary_keywords[:5]) if len(secondary_keywords) > 5 else ', '.join(secondary_keywords)}" if secondary_keywords else ""}
   
   {f"Limit each secondary keyword to maximum 2 mentions in the entire content." if secondary_keywords else ""}

9. VERY IMPORTANT: This is chunk of a larger HTML document. Translate ONLY what's provided. Don't try to complete or start tags that seem incomplete - they will be joined with other chunks.

       Return exactly a valid JSON object matching the following schema without any additional text:
       {{
         "translated_html": "STRING",
         "titles": ["STRING"],
         "meta_descriptions": ["STRING"],
         "alt_text": "STRING",
         "wordpress_slug": "STRING"
       }}

       Title best practices (apply these when creating titles and provide 3 Thai title options in the "titles" array):
       - Place the primary keyword at or near the beginning for stronger relevance
       - Keep length concise (about 50–60 Thai characters) to avoid truncation
       - Use a number or the year 2025 when it feels natural to boost CTR

       For meta_descriptions, create 3 distinct meta descriptions that:
       - Each contains the primary keyword once
       - Each contains 2-3 different secondary keywords
       - Are between 150-160 characters in length
       - Have different sentence structures and focuses

The wordpress_slug should be an English-language URL-friendly version that:
- Closely matches high-volume search terms in your market
- Is concise (3-5 words maximum)
- Includes the year (2025) and main topic (crypto/altcoin)
- Focuses on the investment aspect (e.g., "best-crypto-investments-2025")

Content to translate:
{text}

Return ONLY the JSON object, with no extra text or commentary."""
    
    client_config = init_openrouter_client()
    
    for attempt in range(max_retries):
        st.info(f"Chunk processing attempt {attempt + 1}/{max_retries}...") # Log attempt start
        try:
            st.info(f"Making API request (Timeout: {240}s)...") # Log before request
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
            
            st.info(f"API request completed with status: {response.status_code}") # Log after request
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
            st.info("Received API response content.") # Log response received
            
        except requests.exceptions.HTTPError as http_err:
            st.error(f"HTTP Error: {http_err} - Attempt {attempt + 1}/{max_retries}")
            # Log the response content if available, as it might contain useful error details
            try:
                error_details = response.json()
                st.error(f"API Error Details: {error_details}")
            except json.JSONDecodeError:
                st.error(f"API Response Content (non-JSON): {response.text[:500]}") # Log first 500 chars
            
            st.warning(f"Retrying after HTTP error (attempt {attempt + 1})...") # Log before retry
            if attempt == max_retries - 1:
                st.error("Max retries reached after HTTP error.")
                return None
        except requests.exceptions.RequestException as e:
            st.error(f"API Request failed: {str(e)} - Attempt {attempt + 1}/{max_retries}")
            
            st.warning(f"Retrying after request exception (attempt {attempt + 1})...") # Log before retry
            if attempt == max_retries - 1:
                st.error("Max retries reached after request exception.")
                return None

        if not resp_text:
            st.error(f"Empty response from API. Attempt {attempt+1}/{max_retries}")
            
            st.warning(f"Retrying after empty response (attempt {attempt + 1})...") # Log before retry
            if attempt == max_retries - 1:
                st.error("Max retries reached after empty response.")
                return None

        st.info("Attempting to extract JSON...") # Log before extraction
        result = extract_json_safely(resp_text)
        if result is None:
            st.error(f"Failed to extract JSON on attempt {attempt+1}/{max_retries}. Preview: {st.session_state.get('debug_log', 'N/A')}")
            
            st.warning(f"Retrying after JSON extraction failure (attempt {attempt + 1})...") # Log before retry
            if attempt == max_retries - 1:
                st.error("Max retries reached after JSON extraction failure.")
                return None

        st.info("Checking for missing fields...") # Log before field check
        required = ['translated_html', 'titles', 'meta_descriptions', 'alt_text']
        optional = ['wordpress_slug']
        
        missing = [field for field in required if field not in result]
        if missing:
            st.error(f"Missing required fields: {missing} Attempt {attempt+1}/{max_retries}")
            
            st.warning(f"Retrying after missing fields (attempt {attempt + 1})...") # Log before retry
            if attempt == max_retries - 1:
                st.error("Max retries reached after missing fields.")
                return None
            
        # Add wordpress_slug if missing
        if 'wordpress_slug' not in result and result.get('titles') and result['titles']:
            # Create a simple slug from the first title
            from re import sub
            title = result['titles'][0]
            result['wordpress_slug'] = sub(r'[^\w\s-]', '', title.lower())
            result['wordpress_slug'] = sub(r'[\s-]+', '-', result['wordpress_slug'])

        result['translated_html'] = html.unescape(result['translated_html'])
        st.success(f"Chunk processing successful on attempt {attempt + 1}.") # Log success
        return result
        
    st.error("Chunk processing failed after all retries.") # Log overall failure
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

def translate_chunk_with_fallback(chunk, primary_keyword="", secondary_keywords=None, max_level=3):
    """
    Attempts to translate a chunk using _translate_single.
    If translation fails and max_level is not exceeded, the chunk is split intelligently.
    Returns a dictionary with combined translated_html and aggregated SEO elements or None if translation fails.
    """
    secondary_keywords = secondary_keywords or []
    
    # First attempt: try to translate the whole chunk
    result = _translate_single(chunk, primary_keyword, secondary_keywords)
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
    result1 = translate_chunk_with_fallback(chunk1, primary_keyword, secondary_keywords, max_level-1)
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

def translate_content(content: str, primary_keyword: str = "", secondary_keywords: list = None, max_workers: int = 1) -> dict:
    """
    Translate HTML content. If content is too long, split it into chunks,
    translate each using fallback logic, and combine results.
    """
    secondary_keywords = secondary_keywords or []
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
    
    if total_length <= CHUNK_LIMIT:
        result = translate_chunk_with_fallback(content, primary_keyword, secondary_keywords)
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
            return _translate_single(chunk, chunk_primary, secondary_keywords)

        with ThreadPoolExecutor(max_workers=max(1, int(max_workers))) as executor:
            future_map = {executor.submit(submit_task, i, chunk): i for i, chunk in enumerate(chunks)}
            for future in as_completed(future_map):
                i = future_map[future]
                try:
                    result = future.result()
                except Exception as e:
                    result = None
                    st.error(f"Chunk {i+1} failed with exception: {e}")
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

# Initialize translation history cache and debug log
if 'history' not in st.session_state:
    st.session_state['history'] = []
    
if 'debug_log' not in st.session_state:
    st.session_state['debug_log'] = ""

# Initialize OpenRouter client
try:
    openrouter_config = init_openrouter_client()
    st.sidebar.success(f"Using model: {openrouter_config['default_model']}")
except Exception as e:
    st.error(f"Failed to initialize OpenRouter client: {str(e)}")
    st.stop()

# Streamlit UI
st.title("Crypto HTML Translation")
st.subheader("Thai Crypto Journalist Translation Tool")

upload_method = st.radio("Choose input method:", ["Text Input", "File Upload"])
html_input = ""
if upload_method == "Text Input":
    html_input = st.text_area("Enter HTML content to translate:", height=200)
else:
    uploaded_file = st.file_uploader("Upload HTML file", type=['html', 'htm', 'txt'])
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

with st.expander("Advanced Options"):
    st.info("These settings help optimize translation of very large documents.")
    save_intermediate = st.checkbox("Save intermediate results (recommended for large files)", value=True)
    concurrency = st.slider("Parallel requests (higher is faster but may hit rate limits)", min_value=1, max_value=8, value=3)
    clear_cache = st.button("Clear cache")
    
    if clear_cache:
        cache_keys = [k for k in st.session_state.keys() if k.startswith('translation_cache_')]
        for key in cache_keys:
            del st.session_state[key]
        st.success("Cache cleared!")

if st.button("Translate"):
    if not html_input:
        st.error("Please enter HTML content to translate")
    else:
        line_count = html_input.count('\n') + 1
        char_count = len(html_input)
        tag_count = len(re.findall(r'<[^>]+>', html_input))
        
        st.info(f"Document stats: {line_count} lines, {char_count} characters, approximately {tag_count} HTML tags")
        
        start_time = time.time()
        with st.spinner("Analyzing document..."):
            try:
                result = translate_content(html_input, primary_keyword, secondary_keywords, max_workers=concurrency)
            except Exception as e:
                st.error(f"Translation failed: {str(e)}")
                result = None

            end_time = time.time()
            process_time = end_time - start_time

            if result:
                st.success(f"Translation complete in {process_time:.1f} seconds!")
                # Output vs Input comparison (rows/lines, characters, and HTML tags)
                out_html = result['translated_html']
                out_line_count = out_html.count('\n') + 1
                out_char_count = len(out_html)
                out_tag_count = len(re.findall(r'<[^>]+>', out_html))

                # Avoid division by zero
                line_ratio = (out_line_count / line_count) if line_count else 1.0
                char_ratio = (out_char_count / char_count) if char_count else 1.0
                tag_ratio = (out_tag_count / tag_count) if tag_count else 1.0

                st.info(
                    f"Output stats: {out_line_count} lines, {out_char_count} characters, approximately {out_tag_count} HTML tags\n"
                    f"Ratios vs input → Lines: {line_ratio:.2f}×, Chars: {char_ratio:.2f}×, Tags: {tag_ratio:.2f}×"
                )

                # Shortcode and structure checks
                shortcode_pattern = r"\[[^\[\]]+\]"
                in_shortcodes = re.findall(shortcode_pattern, html_input)
                out_shortcodes = re.findall(shortcode_pattern, out_html)
                if in_shortcodes:
                    st.info(f"Shortcodes: input {len(in_shortcodes)} → output {len(out_shortcodes)}")
                    if len(out_shortcodes) < len(in_shortcodes):
                        st.warning("⚠️ Some shortcodes may be missing in the output.")

                # Warn if output appears significantly shorter than input
                # Use chars/tags as stronger signal than lines (which can change due to formatting)
                if (char_ratio < 0.90) or (tag_ratio < 0.90):
                    st.warning(
                        "⚠️ The translated output is significantly smaller by characters or tag count. "
                        "This could indicate content omission by the model."
                    )
                
                history_entry = {
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'input_length': len(html_input),
                    'output_length': len(result['translated_html']),
                    'keyword': primary_keyword if primary_keyword else "None",
                    'process_time': process_time
                }
                st.session_state['history'].append(history_entry)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        label="Download HTML",
                        data=result['translated_html'],
                        file_name="translated.html",
                        mime="text/html"
                    )
                with col2:
                    json_data = json.dumps(result, ensure_ascii=False, indent=2)
                    st.download_button(
                        label="Download JSON (All Results)",
                        data=json_data,
                        file_name="translation_results.json",
                        mime="application/json"
                    )
                
                with st.expander("Translated HTML Content", expanded=True):
                    st.code(result['translated_html'], language='html')
                
                with st.expander("SEO Elements", expanded=True):
                    st.write("Titles:", result['titles'])
                    st.write("Meta Descriptions:", result['meta_descriptions'])
                    st.write("Alt Text:", result['alt_text'])
                    st.write("WordPress Slug:", result.get('wordpress_slug', ''))
                
                # Keyword usage analysis
                if primary_keyword:
                    primary_count = result['translated_html'].lower().count(primary_keyword.lower())
                    if primary_count == 0:
                        st.warning(f"⚠️ Primary keyword '{primary_keyword}' is missing from the translation!")
                    else:
                        st.info(f"✅ Primary keyword '{primary_keyword}' appears {primary_count} times in the translated content.")
                
                # Check for missing secondary keywords
                missing_keywords = []
                for sec_keyword in secondary_keywords:
                    sec_count = result['translated_html'].lower().count(sec_keyword.lower())
                    if sec_count == 0:
                        missing_keywords.append(sec_keyword)
                        st.warning(f"⚠️ Secondary keyword '{sec_keyword}' is missing from the translation!")
                    else:
                        st.info(f"✅ Secondary keyword '{sec_keyword}' appears {sec_count} times in the translated content.")
                
                # Summary of keyword coverage
                if secondary_keywords:
                    coverage = ((len(secondary_keywords) - len(missing_keywords)) / len(secondary_keywords)) * 100
                    if coverage == 100:
                        st.success(f"🎯 All keywords successfully integrated in the translation!")
                    else:
                        st.info(f"Keyword coverage: {coverage:.1f}% ({len(secondary_keywords) - len(missing_keywords)}/{len(secondary_keywords)} keywords)")

with st.sidebar.expander("Translation History"):
    if not st.session_state['history']:
        st.write("No translations yet")
    else:
        for i, entry in enumerate(st.session_state['history']):
            st.write(f"**{entry['timestamp']}**")
            st.write(f"Size: {entry['input_length']} → {entry['output_length']} chars")
            st.write(f"Keyword: {entry['keyword']}")
            st.write(f"Time: {entry['process_time']:.1f} seconds")
            st.write("---")
