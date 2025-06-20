import streamlit as st
import os
import json
import html
import re
import time
import requests
import hashlib
from io import StringIO
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

def _optimize_single(text: str, primary_keyword: str = "", secondary_keywords: list = None, max_retries=3, simplified_mode=False) -> dict:
    """Optimize a single chunk of Thai HTML content using Claude via OpenRouter with retries."""
    secondary_keywords = secondary_keywords or []
    
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
   Do not modify any technical elements such as URLs, placeholders (e.g., [cur_year], {{{{variable}}}}), and shortcodes (e.g., [su_note], [toc]). They must remain exactly as they are.

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
        result2 = optimize_chunk_with_fallback(chunk2, "", secondary_keywords, max_level-1)
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

def optimize_content(content: str, primary_keyword: str = "", secondary_keywords: list = None) -> dict:
    """
    Optimize Thai HTML content for keywords. If content is too long, split it into chunks,
    optimize each using fallback logic, and combine results.
    Ensures adequate primary keyword density throughout the content.
    Compares keyword occurrences before and after optimization.
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
            
            result = optimize_chunk_with_fallback(chunk, chunk_primary, secondary_keywords)
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
                result = optimize_content(html_input, primary_keyword, secondary_keywords)
            except Exception as e:
                st.error(f"Optimization failed: {str(e)}")
                result = None

            end_time = time.time()
            process_time = end_time - start_time

            if result:
                st.success(f"Optimization complete in {process_time:.1f} seconds!")
                
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
