import streamlit as st
import re
import json
import logging
import os
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

# Import necessary functions from search2.py
from search2 import (
    init_openrouter_client,
    parse_manual_links,
    add_internal_links_to_html,
    make_llm_request_for_html_linking,
    clean_html_content,
    prepare_html_for_streamlit
)

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    st.set_page_config(
        page_title="HTML Link Enhancer", 
        layout="wide", 
        initial_sidebar_state="collapsed"
    )
    
    # Custom CSS for the enhancement section
    st.markdown("""
    <style>
    .html-enhancement-section {
        background-color: #f0f8ff;
        padding: 20px;
        border-radius: 10px;
        border: 2px solid #4CAF50;
        margin-bottom: 30px;
    }
    .cta-section {
        margin-top: 30px;
        margin-bottom: 20px;
        padding: 15px;
        border: 1px solid #e0e0e0;
        border-radius: 5px;
        background-color: #f8f9fa;
    }
    .cta-section ul {
        list-style-type: none;
        padding-left: 10px;
    }
    .cta-section li {
        margin-bottom: 10px;
    }
    .cta-section a {
        color: #1E88E5;
        font-weight: bold;
        text-decoration: none;
    }
    .cta-section a:hover {
        text-decoration: underline;
    }
    .cta-button-wrapper {
        margin: 15px 0;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("🔗 HTML Link Enhancer")    
    # Initialize LLM client
    gemini_client = init_openrouter_client()
    if not gemini_client:
        st.error("Failed to initialize LLM client. Check API keys and configurations.")
        st.info("Make sure you have OPENROUTER_API_KEY set in your .env file.")
        return

    # Main enhancement section
    
    col1, col2 = st.columns(2)
    
    with col1:
        existing_html_input = st.text_area(
            "📄 Paste Your Existing HTML Content:",
            height=400,
            key="existing_html_input",
            help="Paste your existing HTML article content here. The AI will analyze it and add internal links intelligently."
        )
    
    with col2:
        direct_links_input = st.text_area(
            "🔗 Internal Link Candidates (Title/URL pairs):",
            height=400,
            key="direct_links_input",
            help="Example:\nTrump และ GameStop แห่ถือ Bitcoin! กลยุทธ์ Saylor เขย่าตลาด\nhttps://cryptodnes.bg/th/donald-trump-gamestop-bitcoin-strategy-saylor/"
        )
    
    # Enhancement button
    if st.button("🚀 Enhance HTML with Internal Links", type="primary", key="enhance_html_button"):
        if not existing_html_input.strip():
            st.error("Please paste your existing HTML content first.")
        elif not direct_links_input.strip():
            st.error("Please provide internal link candidates.")
        else:
            # Set timestamp for unique filenames
            import datetime
            st.session_state['timestamp'] = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Check content length and warn if very large
            content_length = len(existing_html_input)
            if content_length > 50000:
                st.warning(f"⚠️ Your content is quite large ({content_length:,} characters). Processing may take longer and there's a small chance of truncation with very long articles.")
            
            candidate_links = parse_manual_links(direct_links_input)
            if not candidate_links:
                st.warning("No valid links parsed from the input. Please check the format (Title on one line, URL on the next).")
            else:
                with st.spinner("🤖 AI is analyzing your HTML and inserting relevant internal links..."):
                    enhanced_html = add_internal_links_to_html(
                        existing_html_input, 
                        candidate_links, 
                        {"client": gemini_client["client"], "model": gemini_client["model"]}
                    )
                
                if enhanced_html and enhanced_html != existing_html_input:
                    # Check for potential truncation
                    original_length = len(existing_html_input)
                    enhanced_length = len(enhanced_html)
                    
                    if enhanced_length < original_length * 0.8:  # If enhanced is significantly shorter
                        st.warning(f"⚠️ Potential truncation detected! Original: {original_length:,} chars, Enhanced: {enhanced_length:,} chars. The AI may have truncated your content.")
                        st.info("💡 Try breaking your content into smaller sections and processing them separately.")
                    else:
                        st.success("✅ Internal links successfully added!")
                    
                    # Show before/after comparison
                    st.markdown("### 📊 Results")
                    
                    # Count links before and after
                    original_link_count = len(re.findall(r'<a[^>]*href=', existing_html_input))
                    enhanced_link_count = len(re.findall(r'<a[^>]*href=', enhanced_html))
                    links_added = enhanced_link_count - original_link_count
                    
                    st.info(f"📈 Added {links_added} internal links to your content (from {original_link_count} to {enhanced_link_count} total links)")
                    
                    # Display enhanced HTML for preview
                    with st.expander("👀 Preview Enhanced Content", expanded=True):
                        streamlit_ready_html = prepare_html_for_streamlit(enhanced_html)
                        st.markdown(streamlit_ready_html, unsafe_allow_html=True)
                    
                    # Provide clean HTML for copying
                    st.markdown("### 📋 Copy Enhanced HTML")
                    cleaned_enhanced_html = clean_html_content(enhanced_html)
                    st.text_area(
                        "Enhanced HTML (ready for WordPress):", 
                        cleaned_enhanced_html, 
                        height=400, 
                        key="enhanced_html_output",
                        help="Copy this enhanced HTML and paste it into your WordPress editor or wherever you need it."
                    )
                    
                elif enhanced_html == existing_html_input:
                    st.warning("⚠️ The AI did not make any changes. This might happen if it couldn't find relevant places to insert the provided links, or if the links don't match the content semantically.")
                    st.info("💡 Try providing more relevant links or check that your link titles match the content topics.")
                else:
                    st.error("❌ Failed to enhance HTML with internal links. Please try again.")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Instructions section
    st.markdown("---")
    st.markdown("## 📖 How to Use")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 📝 Step 1: Prepare Your HTML
        - Copy HTML content from your article
        - Can be from WordPress, other CMS, or any HTML source
        - The content should be well-structured with headings and paragraphs
        """)
        
        st.markdown("""
        ### 🔗 Step 2: Add Link Candidates
        - Format: Title on one line, URL on the next
        - Provide relevant internal links you want to add
        - The AI will determine the best placement
        """)
    
    with col2:
        st.markdown("""
        ### 🤖 Step 3: Let AI Enhance
        - Click the enhancement button
        - AI analyzes content semantically
        - Creates natural transitional paragraphs
        - Preserves all original content
        """)
        
        st.markdown("""
        ### 📋 Step 4: Copy & Use
        - Preview the enhanced content
        - Copy the clean HTML output
        - Paste into WordPress or any platform
        """)
    
    # Example section
    st.markdown("---")
    st.markdown("## 💡 Example Input Format")
    
    with st.expander("View Example Link Candidates Format", expanded=False):
        st.code("""Trump และ GameStop แห่ถือ Bitcoin! กลยุทธ์ Saylor เขย่าตลาด
https://cryptodnes.bg/th/donald-trump-gamestop-bitcoin-strategy-saylor/

7 เหรียญคริปโตที่น่าลงทุน 2025 ที่อาจพลิกเงินหลักพันเป็นแสนบาทได้!
https://cryptodnes.bg/th/7-altcoins-to-watch-2025/

Bitcoin ETF ทุบสถิติ! BlackRock นำทีมสถาบันเทขายหุ้นซื้อ BTC
https://cryptodnes.bg/th/bitcoin-etf-blackrock-institutional-buying/""")
    
    # Footer
    st.markdown("---")
    st.markdown("*Powered by AI-driven semantic analysis for intelligent internal linking*")

if __name__ == "__main__":
    main()
