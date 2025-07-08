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

# Configure comprehensive logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('optimizer_debug.log')
    ]
)
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
        raise ValueError("OpenRouter API key not found")
    
    return {
        'api_key': api_key,
        'default_model': "anthropic/claude-sonnet-4",
        'api_url': "https://openrouter.ai/api/v1/chat/completions"
    }

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

def _optimize_single(text: str, primary_keyword: str = "", secondary_keywords: list = None, max_retries=3, simplified_mode=False, target_primary_density=2.5) -> dict:
    """Optimize a single chunk of Thai HTML content using Claude via OpenRouter with retries."""
    secondary_keywords = secondary_keywords or []
    
    logging.debug(f"[OPTIMIZE] Starting optimization of chunk: {len(text)} chars, simplified_mode={simplified_mode}")
    logging.debug(f"[OPTIMIZE] Primary keyword: '{primary_keyword}', Secondary keywords: {len(secondary_keywords)}")
    logging.debug(f"[OPTIMIZE] Target primary keyword density: {target_primary_density}%")
    
    # Calculate target keyword counts based on content length
    word_count = len(text.split())
    target_primary_count = max(3, int(word_count * target_primary_density / 100)) if primary_keyword else 0
    target_secondary_count = max(2, int(word_count * 1.5 / 100)) if secondary_keywords else 0
    
    logging.debug(f"[OPTIMIZE] Target counts - Primary: {target_primary_count}, Secondary: {target_secondary_count} each")
    
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
        # Use the full detailed prompt for normal optimization
        prompt = f"""You are a Senior Thai SEO specialist focusing on crypto content. Optimize this Thai HTML content for SEO by following these strict requirements:

1. Preserve HTML Structure
   Keep all HTML tags, attributes, and inline styles completely unchanged. Do not modify the HTML in any way except to optimize the visible Thai text.

2. Retain Technical Content
   Do not modify any technical elements such as URLs, placeholders (e.g., [cur_year], {{{{variable}}}}), and shortcodes (e.g., [su_note], [toc]). They must remain exactly as they are.

3. Keep Entity Names (people, places, organisation), Brands, coin names, and Technical Terms in English.

4. Optimize Content Without Changing Meaning
   You must maintain the meaning and intent of the original Thai content. Your goal is to optimize the text for keywords while keeping the original meaning intact. Do not add new information that wasn't in the original content.

5. Ensure Proper Thai Spacing and Punctuation
   Optimize the content while strictly following Thai orthographic and typographic conventions. Make sure no extra spaces or punctuation errors are introduced.

6. AGGRESSIVE Keyword Integration Strategy:
   Primary Keyword: {primary_keyword if primary_keyword else "None provided"}
   {f"CRITICAL: You MUST integrate the primary keyword '{primary_keyword}' at least {target_primary_count} times throughout this content chunk. Target density: {target_primary_density}%. Place the keyword in:" if primary_keyword else "No primary keyword provided."}
   {f"   - Headings (H1, H2, H3) where contextually appropriate" if primary_keyword else ""}
   {f"   - First paragraph (mandatory)" if primary_keyword else ""}
   {f"   - Multiple body paragraphs naturally integrated" if primary_keyword else ""}
   {f"   - List items and bullet points where relevant" if primary_keyword else ""}
   {f"   - Image alt text and captions if present" if primary_keyword else ""}
   {f"   - Meta descriptions" if primary_keyword else ""}
   {f"   - Use keyword variations and synonyms to avoid over-optimization" if primary_keyword else ""}
   
   Secondary Keywords: {', '.join(secondary_keywords) if secondary_keywords else "None provided"}
   {f"IMPORTANT: You MUST include EACH secondary keyword at least {target_secondary_count} times in this content chunk. Distribute them across:" if secondary_keywords else "No secondary keywords provided."}
   {f"   - H2 and H3 headings (prioritize these)" if secondary_keywords else ""}
   {f"   - Body paragraphs with natural integration" if secondary_keywords else ""}
   {f"   - FAQ sections if present" if secondary_keywords else ""}
   {f"   - List items and descriptions" if secondary_keywords else ""}
   {f"   - Image captions and alt text" if secondary_keywords else ""}
   
   {f"Priority secondary keywords (focus on these first): {', '.join(secondary_keywords[:3])}" if len(secondary_keywords) > 3 else f"Secondary keywords to integrate: {', '.join(secondary_keywords)}" if secondary_keywords else ""}
   
   KEYWORD DENSITY TARGETS:
   - Primary keyword: {target_primary_density}% density (approximately {target_primary_count} mentions)
   - Each secondary keyword: 1.5% density (approximately {target_secondary_count} mentions each)
   - Use natural variations, synonyms, and related terms
   - Ensure keywords flow naturally within sentences

7. VERY IMPORTANT: This is chunk of a larger HTML document. Optimize ONLY what's provided. Don't try to complete or start tags that seem incomplete - they will be joined with other chunks.

Return exactly a valid JSON object matching the following schema without any additional text:
{{
  "optimized_html": "STRING",
  "titles": ["STRING"],
  "meta_descriptions": ["STRING"],
  "alt_text": "STRING",
  "wordpress_slug": "STRING"
}}

For meta_descriptions, create 3 distinct meta descriptions that:
- Each contains the primary keyword at least once
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
            payload = json.dumps({
                "model": client_config['default_model'],
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            })
            
            logging.debug(f"[OPTIMIZE] Sending request to OpenRouter API with payload size: {len(payload)} chars")
            
            response = requests.post(
                url=client_config['api_url'],
                headers={
                    "Authorization": f"Bearer {client_config['api_key']}",
                    "Content-Type": "application/json"
                },
                data=payload,
                timeout=240
            )
            
            logging.debug(f"[OPTIMIZE] API response status: {response.status_code}")
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
            logging.debug(f"[OPTIMIZE] Response text length: {len(resp_text)} chars")
            
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

def optimize_chunk_with_fallback(chunk, primary_keyword="", secondary_keywords=None, max_level=3):
    """
    Attempts to optimize a chunk using _optimize_single.
    If optimization fails and max_level is not exceeded, the chunk is split into halves recursively.
    Returns a dictionary with combined optimized_html and aggregated SEO elements or None if optimization fails.
    """
    secondary_keywords = secondary_keywords or []
    result = _optimize_single(chunk, primary_keyword, secondary_keywords)
    if result is not None:
        return result
    else:
        if max_level <= 0 or len(chunk) < 1000:
            return None
        mid = len(chunk) // 2
        chunk1 = chunk[:mid]
        chunk2 = chunk[mid:]
        result1 = optimize_chunk_with_fallback(chunk1, primary_keyword, secondary_keywords, max_level-1)
        result2 = optimize_chunk_with_fallback(chunk2, primary_keyword, secondary_keywords, max_level-1)
        if result1 is None or result2 is None:
            return None
        combined_html = result1['optimized_html'] + result2['optimized_html']
        titles = result1['titles'] + result2['titles']
        meta_descriptions = result1['meta_descriptions'] + result2['meta_descriptions']
        alt_text = result1['alt_text'] if result1['alt_text'].strip() else result2['alt_text']
        wordpress_slug = result1.get('wordpress_slug', '') if result1.get('wordpress_slug', '').strip() else result2.get('wordpress_slug', '')
        return {
            'optimized_html': combined_html,
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

def optimize_content(content: str, primary_keyword: str = "", secondary_keywords: list = None) -> dict:
    """
    Optimize Thai HTML content for keywords. If content is too long, split it into chunks,
    optimize each using fallback logic, and combine results.
    """
    secondary_keywords = secondary_keywords or []
    content_hash = hashlib.md5(content.encode()).hexdigest()
    keywords_hash = hashlib.md5(f"{primary_keyword}{''.join(secondary_keywords)}".encode()).hexdigest()
    cache_key = f"optimization_cache_{content_hash}_{keywords_hash}"
    cached_result = st.session_state.get(cache_key)
    if cached_result:
        st.success("Retrieved from cache!")
        return cached_result
    
    total_length = len(content)
    if total_length < 10000:
        CHUNK_LIMIT = 6000  # Reduced from 8000
    elif total_length < 50000:
        CHUNK_LIMIT = 12000  # Reduced from 20000
    else:
        CHUNK_LIMIT = 10000  # Significantly reduced from 30000
    
    if total_length <= CHUNK_LIMIT:
        result = optimize_chunk_with_fallback(content, primary_keyword, secondary_keywords)
        if result:
            st.session_state[cache_key] = result
        return result
    else:
        logging.debug(f"[CHUNKING] Starting chunking process for {total_length} chars with limit {CHUNK_LIMIT}")
        chunks = smart_chunk_html(content, CHUNK_LIMIT)
        actual_chunks = len(chunks)
        
        # Log chunk information
        total_chunk_chars = sum(len(chunk) for chunk in chunks)
        logging.debug(f"[CHUNKING] Created {actual_chunks} chunks, total chars: {total_chunk_chars}")
        for i, chunk in enumerate(chunks):
            logging.debug(f"[CHUNKING] Chunk {i+1}: {len(chunk)} chars")
        
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
            
            logging.debug(f"[CHUNK_PROCESS] Processing chunk {i+1}/{actual_chunks}: {len(chunk)} chars")
            
            # Distribute primary keyword across ALL chunks for maximum keyword density
            chunk_primary = primary_keyword if primary_keyword else ""
            logging.debug(f"[CHUNK_PROCESS] Chunk {i+1} will use primary keyword: '{chunk_primary}'")
            
            result = optimize_chunk_with_fallback(chunk, chunk_primary, secondary_keywords)
            if result is None:
                logging.error(f"[CHUNK_PROCESS] Chunk {i+1} optimization FAILED")
                failed_chunks.append(i)
                continue
            
            optimized_length = len(result['optimized_html'])
            logging.debug(f"[CHUNK_PROCESS] Chunk {i+1} optimization SUCCESS: {len(chunk)} → {optimized_length} chars")
            
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
        
        # Final assembly with detailed logging
        logging.debug(f"[ASSEMBLY] Combining {len(optimized_chunks)} optimized chunks")
        total_optimized_chars = sum(len(chunk) for chunk in optimized_chunks)
        logging.debug(f"[ASSEMBLY] Total optimized chars before joining: {total_optimized_chars}")
        
        combined_html = "".join(optimized_chunks)
        logging.debug(f"[ASSEMBLY] Final combined HTML length: {len(combined_html)} chars")
        
        # Calculate and log completion ratio
        completion_ratio = len(combined_html) / total_length if total_length > 0 else 0
        logging.info(f"[ASSEMBLY] Completion ratio: {completion_ratio:.1%} ({len(combined_html)}/{total_length} chars)")
        
        if completion_ratio < 0.5:
            logging.warning(f"[ASSEMBLY] SEVERE TRUNCATION DETECTED: Only {completion_ratio:.1%} of original content retained")
        elif completion_ratio < 0.8:
            logging.warning(f"[ASSEMBLY] MODERATE TRUNCATION: {completion_ratio:.1%} of original content retained")
        
        final_result = {
            'optimized_html': combined_html,
            'titles': seo_elements['titles'],
            'meta_descriptions': seo_elements['meta_descriptions'],
            'alt_text': seo_elements['alt_text'],
            'wordpress_slug': seo_elements['wordpress_slug']
        }
        st.session_state[cache_key] = final_result
        return final_result

def insert_internal_links(html_content: str, internal_links: list, max_retries=3) -> str:
    """
    Step 2: Insert internal links into already optimized HTML content.
    This is a separate, focused operation that happens after SEO optimization.
    """
    if not internal_links or not html_content:
        return html_content
    
    # Prepare internal links for the prompt
    links_text = ""
    for i, link in enumerate(internal_links[:3], 1):  # Limit to 3 links
        links_text += f"{i}. Title: {link['title']}\n   URL: {link['url']}\n"
    
    prompt = f"""You are a content editor specializing in internal link placement. Your task is to insert internal links naturally into this HTML content.

INSTRUCTIONS:
1. Insert 2-3 internal links from the list below into relevant sections of the content
2. Place links where they naturally fit with the content topic and flow
3. Use the article title as anchor text (or a relevant portion of it)
4. DO NOT place links in headings (H1, H2, H3, etc.) or image captions
5. DO NOT modify the URLs - use them exactly as provided
6. Format links as: <a href="[exact-url]">[title-or-relevant-text]</a>
7. Preserve all existing HTML structure and content
8. Return ONLY the modified HTML content, no additional text or commentary

AVAILABLE INTERNAL LINKS:
{links_text}

HTML CONTENT TO MODIFY:
{html_content}

Return only the HTML with internal links inserted:"""

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
                timeout=120
            )
            
            response.raise_for_status()
            resp_data = response.json()
            
            if response.status_code == 429:
                wait_time = int(response.headers.get('Retry-After', 30))
                st.warning(f"Rate limited during link insertion. Waiting {wait_time} seconds...")
                time.sleep(wait_time)
                continue
            
            if not resp_data or 'choices' not in resp_data or not resp_data['choices']:
                st.warning(f"Invalid response during link insertion. Attempt {attempt+1}/{max_retries}")
                if attempt == max_retries - 1:
                    return html_content  # Return original if all attempts fail
                time.sleep(2)
                continue
                
            result_html = resp_data['choices'][0]['message']['content'].strip()
            
            # Basic validation - ensure we got HTML back
            if '<' in result_html and '>' in result_html:
                return result_html
            else:
                st.warning(f"Link insertion returned non-HTML content. Attempt {attempt+1}/{max_retries}")
                if attempt == max_retries - 1:
                    return html_content
                time.sleep(2)
                continue
                
        except requests.exceptions.RequestException as e:
            st.warning(f"Link insertion API request failed: {str(e)} - Attempt {attempt + 1}/{max_retries}")
            if attempt == max_retries - 1:
                return html_content  # Return original if all attempts fail
            time.sleep(2)
            continue
    
    return html_content  # Fallback to original content

def validate_keyword_density(html_content: str, primary_keyword: str, secondary_keywords: list = None, target_primary_density: float = 2.5) -> dict:
    """
    Validate keyword density in optimized content and provide detailed analysis.
    Returns validation results with recommendations for improvement.
    """
    secondary_keywords = secondary_keywords or []
    
    # Extract text content from HTML for accurate word counting
    if BeautifulSoup:
        soup = BeautifulSoup(html_content, 'html.parser')
        text_content = soup.get_text()
    else:
        # Fallback: simple HTML tag removal
        text_content = re.sub(r'<[^>]+>', ' ', html_content)
    
    # Clean and count words
    words = text_content.split()
    total_words = len(words)
    
    validation_results = {
        'total_words': total_words,
        'primary_keyword': {
            'keyword': primary_keyword,
            'count': 0,
            'density': 0.0,
            'target_density': target_primary_density,
            'meets_target': False
        },
        'secondary_keywords': [],
        'overall_score': 0
    }
    
    if primary_keyword:
        primary_count = html_content.lower().count(primary_keyword.lower())
        primary_density = (primary_count / total_words * 100) if total_words > 0 else 0
        
        validation_results['primary_keyword'].update({
            'count': primary_count,
            'density': primary_density,
            'meets_target': primary_density >= target_primary_density * 0.8  # Allow 20% tolerance
        })
    
    for keyword in secondary_keywords:
        sec_count = html_content.lower().count(keyword.lower())
        sec_density = (sec_count / total_words * 100) if total_words > 0 else 0
        target_sec_density = 1.5
        
        validation_results['secondary_keywords'].append({
            'keyword': keyword,
            'count': sec_count,
            'density': sec_density,
            'target_density': target_sec_density,
            'meets_target': sec_density >= target_sec_density * 0.7  # Allow 30% tolerance
        })
    
    # Calculate overall score
    primary_score = 1 if validation_results['primary_keyword']['meets_target'] else 0
    secondary_score = sum(1 for kw in validation_results['secondary_keywords'] if kw['meets_target'])
    total_keywords = 1 + len(secondary_keywords) if primary_keyword else len(secondary_keywords)
    
    validation_results['overall_score'] = ((primary_score + secondary_score) / total_keywords * 100) if total_keywords > 0 else 0
    
    return validation_results

def optimize_and_insert_links(content: str, primary_keyword: str = "", secondary_keywords: list = None, internal_links: list = None) -> dict:
    """
    Two-step process: 
    1. SEO optimization (proven working code)
    2. Internal link insertion (separate, focused operation)
    """
    secondary_keywords = secondary_keywords or []
    internal_links = internal_links or []
    
    # Step 1: SEO Optimization
    st.info("🔄 Step 1: Optimizing content for SEO...")
    optimized_result = optimize_content(content, primary_keyword, secondary_keywords)
    
    if not optimized_result:
        st.error("❌ Step 1 failed: SEO optimization unsuccessful")
        return None
    
    st.success("✅ Step 1 complete: SEO optimization successful")
    
    # Step 2: Internal Link Insertion (if links provided)
    if internal_links:
        st.info("🔗 Step 2: Inserting internal links...")
        try:
            final_html = insert_internal_links(optimized_result['optimized_html'], internal_links)
            optimized_result['optimized_html'] = final_html
            
            # Check if links were actually inserted
            inserted_count = 0
            for link in internal_links:
                if link['url'] in final_html:
                    inserted_count += 1
            
            if inserted_count > 0:
                st.success(f"✅ Step 2 complete: {inserted_count} internal links inserted")
            else:
                st.warning("⚠️ Step 2: No internal links were inserted (content may not have suitable placement opportunities)")
                
        except Exception as e:
            st.warning(f"⚠️ Step 2 failed: {str(e)} - Proceeding with SEO-optimized content only")
    else:
        st.info("ℹ️ Step 2 skipped: No internal links provided")
    
    return optimized_result

# Initialize optimization history cache
if 'history' not in st.session_state:
    st.session_state['history'] = []

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
with st.expander("Add Internal Links (Optional)", expanded=False):
    st.info("Add internal links below (Title on one line, URL on the next). These will be inserted AFTER SEO optimization is complete.")
    
    internal_links_input = st.text_area(
        "Internal Links (Title/URL pairs):",
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
                        "url": url
                    })
    
    if manual_internal_links:
        st.success(f"✅ {len(manual_internal_links)} internal links ready for insertion")
        with st.expander("Preview internal links"):
            for i, link in enumerate(manual_internal_links, 1):
                st.write(f"{i}. **{link['title']}**")
                st.write(f"   {link['url']}")

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
        with st.spinner("Processing content..."):
            try:
                # Use the two-step process
                result = optimize_and_insert_links(html_input, primary_keyword, secondary_keywords, manual_internal_links)
            except Exception as e:
                st.error(f"Optimization failed: {str(e)}")
                result = None

            end_time = time.time()
            process_time = end_time - start_time

            if result:
                st.success(f"🎉 Complete! Processed in {process_time:.1f} seconds")
                
                # Calculate completion ratio
                completion_ratio = len(result['optimized_html']) / len(html_input) if len(html_input) > 0 else 0
                if completion_ratio >= 0.9:
                    st.success(f"✅ Content completion: {completion_ratio:.1%} (Excellent)")
                elif completion_ratio >= 0.7:
                    st.info(f"ℹ️ Content completion: {completion_ratio:.1%} (Good)")
                else:
                    st.warning(f"⚠️ Content completion: {completion_ratio:.1%} (May be incomplete)")
                
                history_entry = {
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'input_length': len(html_input),
                    'output_length': len(result['optimized_html']),
                    'keyword': primary_keyword if primary_keyword else "None",
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
                
                # Show internal link insertion summary
                if manual_internal_links:
                    inserted_links = [link['url'] for link in manual_internal_links if link['url'] in result['optimized_html']]
                    if inserted_links:
                        st.success(f"🔗 Successfully inserted {len(inserted_links)} internal links")
                        with st.expander("Internal Links Details"):
                            for i, url in enumerate(inserted_links, start=1):
                                st.write(f"{i}. {url}")
                    else:
                        st.info("ℹ️ No internal links were inserted (content may not have suitable placement opportunities)")
                
                # Enhanced Keyword Density Analysis
                st.info("📊 Analyzing keyword density...")
                validation_results = validate_keyword_density(result['optimized_html'], primary_keyword, secondary_keywords)
                
                with st.expander("📈 Detailed Keyword Density Analysis", expanded=True):
                    st.write(f"**Total Words:** {validation_results['total_words']}")
                    st.write(f"**Overall SEO Score:** {validation_results['overall_score']:.1f}%")
                    
                    if validation_results['overall_score'] >= 80:
                        st.success("🎯 Excellent keyword optimization!")
                    elif validation_results['overall_score'] >= 60:
                        st.info("✅ Good keyword optimization")
                    else:
                        st.warning("⚠️ Keyword optimization needs improvement")
                    
                    # Primary keyword analysis
                    if primary_keyword:
                        pk = validation_results['primary_keyword']
                        st.write("### Primary Keyword Analysis")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Count", pk['count'])
                        with col2:
                            st.metric("Density", f"{pk['density']:.2f}%")
                        with col3:
                            st.metric("Target", f"{pk['target_density']:.1f}%")
                        
                        if pk['meets_target']:
                            st.success(f"✅ Primary keyword '{pk['keyword']}' meets density target")
                        else:
                            st.warning(f"⚠️ Primary keyword '{pk['keyword']}' below target density")
                            recommended_count = int(validation_results['total_words'] * pk['target_density'] / 100)
                            st.info(f"💡 Recommendation: Add {recommended_count - pk['count']} more mentions")
                    
                    # Secondary keywords analysis
                    if secondary_keywords:
                        st.write("### Secondary Keywords Analysis")
                        for sk in validation_results['secondary_keywords']:
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.write(f"**{sk['keyword']}**")
                            with col2:
                                st.write(f"Count: {sk['count']}")
                            with col3:
                                st.write(f"Density: {sk['density']:.2f}%")
                            with col4:
                                if sk['meets_target']:
                                    st.success("✅ Target met")
                                else:
                                    st.warning("⚠️ Below target")
                
                # Legacy keyword analysis for backward compatibility
                if primary_keyword:
                    primary_count = result['optimized_html'].lower().count(primary_keyword.lower())
                    if primary_count == 0:
                        st.warning(f"⚠️ Primary keyword '{primary_keyword}' is missing from the optimized content!")
                    else:
                        st.info(f"✅ Primary keyword '{primary_keyword}' appears {primary_count} times in the optimized content.")
                
                # Check for missing secondary keywords
                missing_keywords = []
                for sec_keyword in secondary_keywords:
                    sec_count = result['optimized_html'].lower().count(sec_keyword.lower())
                    if sec_count == 0:
                        missing_keywords.append(sec_keyword)
                        st.warning(f"⚠️ Secondary keyword '{sec_keyword}' is missing from the optimized content!")
                    else:
                        st.info(f"✅ Secondary keyword '{sec_keyword}' appears {sec_count} times in the optimized content.")
                
                # Summary of keyword coverage
                if secondary_keywords:
                    coverage = ((len(secondary_keywords) - len(missing_keywords)) / len(secondary_keywords)) * 100
                    if coverage == 100:
                        st.success(f"🎯 All keywords successfully integrated in the optimized content!")
                    else:
                        st.info(f"Keyword coverage: {coverage:.1f}% ({len(secondary_keywords) - len(missing_keywords)}/{len(secondary_keywords)} keywords)")

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
