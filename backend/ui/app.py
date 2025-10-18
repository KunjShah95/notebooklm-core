import streamlit as st
import sys
import os
from typing import List, Optional
from PIL import Image
import io

# Add the src directory to the path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.generation.rag import RAGGenerator, MultimodalContent
from src.embeddings.embedding_generator import EmbeddingGenerator
from src.vector_database.milvus_vector_db import MilvusVectorDB

# Page configuration
st.set_page_config(
    page_title="Multimodal RAG Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .citation-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .response-box {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def initialize_components(provider: str, model_name: str):
    """Initialize the RAG components"""
    try:
        # Initialize embedding generator
        embedding_generator = EmbeddingGenerator()

        # Initialize vector database
        vector_db = MilvusVectorDB(
            collection_name="documents",
            embedding_dim=768  # Adjust based on your embedding model
        )

        # Initialize RAG generator
        rag_generator = RAGGenerator(
            embedding_generator=embedding_generator,
            vector_db=vector_db,
            provider=provider,
            model_name=model_name
        )

        return rag_generator
    except Exception as e:
        st.error(f"Failed to initialize components: {str(e)}")
        return None

def process_image_upload(uploaded_file) -> Optional[MultimodalContent]:
    """Process uploaded image file into MultimodalContent"""
    if uploaded_file is None:
        return None

    try:
        # Read the image
        image_bytes = uploaded_file.read()

        # Get MIME type
        mime_type = uploaded_file.type

        return MultimodalContent(
            type="image",
            content=image_bytes,
            mime_type=mime_type,
            metadata={"filename": uploaded_file.name}
        )
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None

def main():
    st.markdown('<h1 class="main-header">🤖 Multimodal RAG Assistant</h1>', unsafe_allow_html=True)
    st.markdown("Ask questions about your documents with support for text and images!")

    # Initialize session state for configuration
    if 'provider' not in st.session_state:
        st.session_state.provider = "groq"
    if 'model_name' not in st.session_state:
        st.session_state.model_name = "llama3-8b-8192"
    if 'max_chunks' not in st.session_state:
        st.session_state.max_chunks = 8
    if 'max_context_chars' not in st.session_state:
        st.session_state.max_context_chars = 4000

    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Provider selection
        provider = st.selectbox(
            "AI Provider",
            ["groq", "ollama"],
            index=["groq", "ollama"].index(st.session_state.provider),
            help="Choose your AI provider"
        )
        st.session_state.provider = provider

        # Model selection based on provider
        if provider == "groq":
            model_options = [
                "llama3-8b-8192",
                "llama3-70b-8192",
                "llama-3.2-11b-vision-instruct",  # Vision model
                "llama-3.2-90b-vision-instruct"   # Vision model
            ]
        else:  # ollama
            model_options = [
                "llama3.2",
                "llava",  # Vision model
                "llava:13b",
                "bakllava",
                "gpt-oss:20b",
                "glm-4.6:cloud",
                "deepseek-r1:8b",
                "gemma3:12b",
                "qwen3:8b",
                "qwen3-vl:235b-cloud",
                "gpt-oss:120b-cloud"
            ]

        model_name = st.selectbox(
            "Model",
            model_options,
            index=model_options.index(st.session_state.model_name) if st.session_state.model_name in model_options else 0,
            help="Choose the model to use"
        )
        st.session_state.model_name = model_name

        # RAG parameters
        st.subheader("RAG Settings")
        max_chunks = st.slider("Max Chunks", 1, 20, st.session_state.max_chunks, help="Maximum number of document chunks to retrieve")
        max_context_chars = st.slider("Max Context (chars)", 1000, 10000, st.session_state.max_context_chars, help="Maximum context length")

        st.session_state.max_chunks = max_chunks
        st.session_state.max_context_chars = max_context_chars

        # Initialize button
        if st.button("🔄 Initialize System", type="primary"):
            with st.spinner("Initializing RAG system..."):
                rag_system = initialize_components(provider, model_name)
                if rag_system:
                    st.session_state.rag_system = rag_system
                    st.success("✅ System initialized successfully!")
                else:
                    st.error("❌ Failed to initialize system")

    # Main content area
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("💬 Ask Your Question")

        # Text input
        query = st.text_area(
            "Enter your question:",
            height=100,
            placeholder="What would you like to know about your documents?"
        )

        # Image upload
        uploaded_image = st.file_uploader(
            "📸 Upload an image (optional)",
            type=["png", "jpg", "jpeg", "gif", "bmp"],
            help="Upload an image to include in your query for multimodal analysis"
        )

        # Document upload
        uploaded_document = st.file_uploader(
            "📄 Upload a document",
            type=["pdf", "txt", "md", "pptx", "docx"],
            help="Upload a document to include in your query for analysis"
        )

        # Show uploaded image preview
        if uploaded_image:
            st.image(uploaded_image, caption="Uploaded Image", width=300)

        # Process uploaded document
        if uploaded_document:
            st.info(f"Uploaded Document: {uploaded_document.name}")
            # Add logic to process the uploaded document here

        # Generate response button
        generate_button = st.button(
            "🚀 Generate Response",
            type="primary",
            disabled=query.strip() == "" or 'rag_system' not in st.session_state
        )

    with col2:
        st.subheader("📊 System Status")

        if 'rag_system' in st.session_state:
            st.success("✅ RAG System Ready")
            st.info(f"**Provider:** {st.session_state.provider}")
            st.info(f"**Model:** {st.session_state.model_name}")
            st.info(f"**Max Chunks:** {st.session_state.max_chunks}")
            st.info(f"**Max Context:** {st.session_state.max_context_chars} chars")
        else:
            st.warning("⚠️ System not initialized")
            st.info("Please configure and initialize the system in the sidebar")

    # Response area
    if generate_button and query.strip():
        with st.spinner("Generating response..."):
            try:
                # Process multimodal content
                multimodal_content = []
                if uploaded_image:
                    image_content = process_image_upload(uploaded_image)
                    if image_content:
                        multimodal_content.append(image_content)

                # Generate response
                result = st.session_state.rag_system.generate_response(
                    query=query,
                    max_chunks=st.session_state.max_chunks,
                    max_context_chars=st.session_state.max_context_chars,
                    multimodal_content=multimodal_content if multimodal_content else None
                )

                # Display response
                st.markdown("---")
                st.subheader("📝 Response")

                with st.container():
                    st.markdown('<div class="response-box">', unsafe_allow_html=True)
                    st.markdown(result.response)
                    st.markdown('</div>', unsafe_allow_html=True)

                # Display citations
                if result.sources_used:
                    st.subheader("📚 Sources Cited")
                    with st.container():
                        st.markdown('<div class="citation-box">', unsafe_allow_html=True)
                        citation_summary = result.get_citation_summary()
                        st.markdown(citation_summary)
                        st.markdown('</div>', unsafe_allow_html=True)

                    # Expandable section for detailed sources
                    with st.expander("🔍 View Detailed Sources"):
                        for i, source in enumerate(result.sources_used, 1):
                            st.markdown(f"**[{i}]** {source.get('source_file', 'Unknown')}")
                            if source.get('content'):
                                st.text_area(
                                    f"Content from source {i}",
                                    source['content'][:500] + "..." if len(source['content']) > 500 else source['content'],
                                    height=100,
                                    disabled=True,
                                    key=f"source_{i}"
                                )

                # Display multimodal content info
                if result.multimodal_content:
                    st.subheader("🖼️ Multimodal Content Processed")
                    for i, content in enumerate(result.multimodal_content):
                        st.info(f"Processed {content.type} content: {content.metadata.get('filename', f'Item {i+1}') if content.metadata else f'Item {i+1}'}")

                # Display metadata
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("Sources Used", result.retrieval_count)
                with col_b:
                    st.metric("Generation Tokens", result.generation_tokens or "N/A")
                with col_c:
                    st.metric("Query Length", len(query))

            except Exception as e:
                st.error(f"❌ Error generating response: {str(e)}")
                st.exception(e)

    # Footer
    st.markdown("---")
    st.markdown("*Built with Streamlit, LiteLLM, and multimodal RAG technology*")

if __name__ == "__main__":
    main()