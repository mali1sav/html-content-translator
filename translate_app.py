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
        'default_model': "anthropic/claude-3.7-sonnet:beta",
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

def _translate_single(text: str, keyword: str = "", max_retries=3) -> dict:
    """Translate a single chunk of HTML content using Claude via OpenRouter with retries."""
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
   - `https://mindofpepe.com` or `https://mindofpepe.com/en` → `https://mindofpepe.com/th`
   - `https://solaxy.io` or `https://solaxy.io/en` → `https://solaxy.io/th`
   - `https://memeindex.com` or `https://memeindex.com/en` → `https://memeindex.com/th`
   - `https://btcbulltoken.com` or `https://btcbulltoken.com/en` → `https://btcbulltoken.com/th`

   Do not alter any other URLs or domains besides these five. All other links must remain exactly as they are in the original HTML.

6. Ensure Proper Spacing and Punctuation
   Translate the visible text content to Thai while strictly following Thai orthographic and typographic conventions. Make sure no extra spaces or punctuation errors are introduced. Insert appropriate spacing where needed between Thai and any English words or technical terms. If there are English words in Thai sentences, ensure they are written in title case. For example: Exchange แบบ Non-Custodial ผู้ใช้จะได้รับ Private Keys ของตน. 

   Avoid word-for-word translation. You must ensure the sentence is correct and resonates with Thai Crypto readers. If needed, you can adjust the position of HTML tags to allow for active (direct) sentence structures rather than passive ones that may originate from the original language sentences.

7. Do not add or remove wrapper tags (like <!DOCTYPE html>, <html>, <head>, or <body>) unless they already exist in the snippet.

8. Keyword to Target: {keyword if keyword else "None provided"}
   {f"Ensure that the keyword '{keyword}' appears at least 7 times where contextually appropriate in the translated text, without violating the above rules." if keyword else "No keyword targeting required."}
   This means if there is a natural spot in the text where you can include the keyword in Thai (or as-is if it's a brand/technical term), do so. Do not force the keyword in places that break grammar or context.

9. VERY IMPORTANT: This is chunk of a larger HTML document. Translate ONLY what's provided. Don't try to complete or start tags that seem incomplete - they will be joined with other chunks.

Return exactly a valid JSON object matching the following schema without any additional text:
{{
  "translated_html": "STRING",
  "titles": ["STRING"],
  "meta_descriptions": ["STRING"],
  "alt_text": "STRING"
}}

Content to translate:
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

        required = ['translated_html', 'titles', 'meta_descriptions', 'alt_text']
        missing = [field for field in required if field not in result]
        if missing:
            st.error(f"Missing required fields: {missing} Attempt {attempt+1}/{max_retries}")
            if attempt == max_retries - 1:
                return None
            time.sleep(2)
            continue

        result['translated_html'] = html.unescape(result['translated_html'])
        return result
    return None

def translate_chunk_with_fallback(chunk, keyword="", max_level=3):
    """
    Attempts to translate a chunk using _translate_single.
    If translation fails and max_level is not exceeded, the chunk is split into halves recursively.
    Returns a dictionary with combined translated_html and aggregated SEO elements or None if translation fails.
    """
    result = _translate_single(chunk, keyword)
    if result is not None:
        return result
    else:
        if max_level <= 0 or len(chunk) < 1000:
            return None
        mid = len(chunk) // 2
        chunk1 = chunk[:mid]
        chunk2 = chunk[mid:]
        result1 = translate_chunk_with_fallback(chunk1, keyword, max_level-1)
        result2 = translate_chunk_with_fallback(chunk2, keyword, max_level-1)
        if result1 is None or result2 is None:
            return None
        combined_html = result1['translated_html'] + result2['translated_html']
        titles = result1['titles'] + result2['titles']
        meta_descriptions = result1['meta_descriptions'] + result2['meta_descriptions']
        alt_text = result1['alt_text'] if result1['alt_text'].strip() else result2['alt_text']
        return {
            'translated_html': combined_html,
            'titles': titles,
            'meta_descriptions': meta_descriptions,
            'alt_text': alt_text
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

def translate_content(content: str, keyword: str = "") -> dict:
    """
    Translate HTML content. If content is too long, split it into chunks,
    translate each using fallback logic, and combine results.
    """
    content_hash = hashlib.md5(content.encode()).hexdigest()
    cache_key = f"translation_cache_{content_hash}_{keyword}"
    cached_result = st.session_state.get(cache_key)
    if cached_result:
        st.success("Retrieved from cache!")
        return cached_result
    
    total_length = len(content)
    if total_length < 10000:
        CHUNK_LIMIT = 8000
    elif total_length < 50000:
        CHUNK_LIMIT = 20000
    else:
        CHUNK_LIMIT = 30000
    
    if total_length <= CHUNK_LIMIT:
        result = translate_chunk_with_fallback(content, keyword)
        if result:
            st.session_state[cache_key] = result
        return result
    else:
        chunks = smart_chunk_html(content, CHUNK_LIMIT)
        actual_chunks = len(chunks)
        progress_bar = st.progress(0)
        status_text = st.empty()
        status_text.text(f"Translating {actual_chunks} chunks...")
        translated_chunks = []
        seo_elements = {
            'titles': [],
            'meta_descriptions': [],
            'alt_text': ""
        }
        keyword_inserted = False
        failed_chunks = []
        for i, chunk in enumerate(chunks):
            progress = i / actual_chunks
            progress_bar.progress(progress)
            status_text.text(f"Translating chunk {i+1}/{actual_chunks}...")
            chunk_keyword = ""
            if keyword and not keyword_inserted:
                chunk_keyword = keyword
                keyword_inserted = True
            result = translate_chunk_with_fallback(chunk, chunk_keyword)
            if result is None:
                failed_chunks.append(i)
                continue
            translated_chunks.append(result['translated_html'])
            if result['titles'] and any(title.strip() for title in result['titles']):
                seo_elements['titles'].extend(result['titles'])
            if result['meta_descriptions'] and any(desc.strip() for desc in result['meta_descriptions']):
                seo_elements['meta_descriptions'].extend(result['meta_descriptions'])
            if result['alt_text'] and result['alt_text'].strip():
                if not seo_elements['alt_text']:
                    seo_elements['alt_text'] = result['alt_text']
            time.sleep(1)
        progress_bar.progress(1.0)
        if failed_chunks:
            status_text.text(f"Warning: {len(failed_chunks)}/{actual_chunks} chunks failed to translate.")
            if not translated_chunks:
                return None
        else:
            status_text.text("Translation complete!")
        combined_html = "".join(translated_chunks)
        final_result = {
            'translated_html': combined_html,
            'titles': seo_elements['titles'],
            'meta_descriptions': seo_elements['meta_descriptions'],
            'alt_text': seo_elements['alt_text']
        }
        st.session_state[cache_key] = final_result
        return final_result

# Initialize translation history cache
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
st.title("Crypto HTML Translation")
st.subheader("Thai Crypto Journalist Translation Tool")

with st.expander("Translation Guidelines", expanded=False):
    st.markdown("""
    **This tool follows these translation guidelines:**
    
    1. **Preserves HTML structure** - All tags and attributes remain intact
    2. **Retains technical content** - URLs, placeholders, and code are preserved
    3. **Maintains English technical terms** - Crypto terminology stays in English
    4. **Translates all visible text** - Including spans and anchor text
    5. **Modifies specific links** - Changes select domains to Thai versions
    6. **Ensures proper Thai spacing and punctuation** .
    7. **Preserves document structure** - No wrapper tags added or removed
    8. **Keyword targeting** - Includes specified keywords naturally in the translation
    """)

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

keyword_input = st.text_input("Target keyword (optional):", placeholder="e.g., web3 wallet")

with st.expander("Advanced Options"):
    st.info("These settings help optimize translation of very large documents.")
    save_intermediate = st.checkbox("Save intermediate results (recommended for large files)", value=True)
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
                result = translate_content(html_input, keyword_input)
            except Exception as e:
                st.error(f"Translation failed: {str(e)}")
                result = None

            end_time = time.time()
            process_time = end_time - start_time

            if result:
                st.success(f"Translation complete in {process_time:.1f} seconds!")
                
                history_entry = {
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'input_length': len(html_input),
                    'output_length': len(result['translated_html']),
                    'keyword': keyword_input if keyword_input else "None",
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
                
                with st.expander("SEO Elements"):
                    st.write("Titles:", result['titles'])
                    st.write("Meta Descriptions:", result['meta_descriptions'])
                    st.write("Alt Text:", result['alt_text'])
                
                if keyword_input:
                    keyword_count = result['translated_html'].lower().count(keyword_input.lower())
                    st.info(f"Keyword '{keyword_input}' appears {keyword_count} times in the translated content.")

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
