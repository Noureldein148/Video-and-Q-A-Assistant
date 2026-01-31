import os
import validators, streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
## Streamlit APP
st.set_page_config(page_title="LangChain: Summarize Text From YT or Website", page_icon="🦜")
st.title("🦜 LangChain: Summarize Text From YT or Website")
st.subheader('Summarize URL')

## Sidebar for Groq configuration
with st.sidebar:
    st.header("� Groq Configuration")

    # Groq API Key input
    groq_api_key = st.text_input(
        "Groq API Key",
        value="",
        type="password",
        help="Enter your Groq API key from console.groq.com",
        placeholder="gsk_..."
    )

    # Model selection for Groq
    model_choice = st.selectbox(
        "Choose Groq Model:",
        [
            "llama-3.1-8b-instant",
            "llama-3.3-70b-versatile",
            "mistral-saba-24b",
            "gemma2-9b-it"
        ],
        index=0,
        help="Select which Groq model to use for summarization"
    )
    
    # Show model info
    model_info = {
        "llama-3.1-8b-instant": "Meta Llama 3.1 8B - Fast and efficient (replaces llama3-8b-8192)",
        "llama-3.3-70b-versatile": "Meta Llama 3.3 70B - Most capable (replaces llama3-70b-8192)",
        "mistral-saba-24b": "Mistral Saba 24B - Good balance of speed and quality (replaces mixtral-8x7b-32768)",
        "gemma2-9b-it": "Google Gemma 2 9B - Latest version"
    }
    
    if model_choice in model_info:
        st.info(f"ℹ️ {model_info[model_choice]}")

generic_url = st.text_input("URL", label_visibility="collapsed", placeholder="Enter YouTube URL or Website URL to summarize")

## Initialize Groq LLM
if groq_api_key:
    llm = ChatGroq(model=model_choice, groq_api_key=groq_api_key)
    st.sidebar.success(f"🚀 Groq model loaded: {model_choice}")
else:
    llm = None
    st.sidebar.warning("⚠️ Please enter your Groq API key")

prompt_template = """
Provide a summary of the following content in 300 words:
Content:{text}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

if st.button("Summarize the Content from YT or Website"):
    ## Validate all the inputs
    if not generic_url.strip():
        st.error("Please provide a URL to get started")
    elif not validators.url(generic_url):
        st.error("Please enter a valid URL. It can be a YouTube video URL or website URL")
    elif llm is None:
        st.error("Please provide your Groq API key to proceed")

    else:
        try:
            with st.spinner(f"Processing with Groq {model_choice}... Please wait..."):
                ## loading the website or yt video data
                if "youtube.com" in generic_url:
                    loader = YoutubeLoader.from_youtube_url(
                        generic_url, add_video_info=False
                    )
                    st.info("📺 Loading YouTube video content...")
                else:
                    loader = UnstructuredURLLoader(urls=[generic_url], ssl_verify=False,
                                                   headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"})
                    st.info("🌐 Loading website content...")

                docs = loader.load()

                if not docs:
                    st.error("❌ No content could be loaded from the URL")
                    st.stop()

                ## Chain For Summarization
                chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                output_summary = chain.run(docs)

                st.success("✅ Summary generated successfully!")
                st.subheader(f"📝 Summary (Generated using Groq {model_choice})")
                st.write(output_summary)

                # Show some metadata
                with st.expander("ℹ️ Document Information"):
                    st.write(f"**Number of documents loaded:** {len(docs)}")
                    st.write(f"**Total characters:** {sum(len(doc.page_content) for doc in docs)}")
                    st.write(f"**Model used:** Groq {model_choice}")
                    if docs and hasattr(docs[0], 'metadata'):
                        st.write(f"**Source:** {docs[0].metadata}")

        except Exception as e:
            st.error(f"❌ Error occurred: {str(e)}")
            st.info("💡 Tips:")
            st.info("• Make sure the URL is accessible")
            st.info("• Check your Groq API key is valid")
            st.info("• Try a different URL if the current one fails")
            st.info("• Some models may have rate limits")