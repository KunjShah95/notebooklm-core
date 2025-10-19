"""
Test script to verify the document upload -> process -> embed -> insert flow
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from ui.app import initialize_components, process_and_index_document
from pathlib import Path

def test_document_indexing():
    """Test the document indexing flow"""
    print("=" * 60)
    print("Testing Document Indexing Flow")
    print("=" * 60)
    
    # Step 1: Initialize components
    print("\n1. Initializing RAG system...")
    rag_system = initialize_components(provider="groq", model_name="llama3-8b-8192")
    
    if not rag_system:
        print("❌ Failed to initialize system")
        return False
    
    print("✅ System initialized successfully")
    print(f"   - Document Processor: {hasattr(rag_system, 'document_processor')}")
    print(f"   - Embedding Generator: {hasattr(rag_system, 'embedding_generator')}")
    print(f"   - Vector DB: {hasattr(rag_system, 'vector_db')}")
    
    # Step 2: Create a test document
    print("\n2. Creating test document...")
    test_doc_path = Path("test_document.txt")
    test_content = """
This is a test document for the Multimodal RAG system.

The system supports multiple document types including:
- PDF files
- Text files
- Markdown files
- PowerPoint presentations
- Word documents

The RAG (Retrieval Augmented Generation) system works by:
1. Processing documents into chunks
2. Generating embeddings for each chunk
3. Storing embeddings in a vector database
4. Retrieving relevant chunks for user queries
5. Generating responses using an LLM with context from retrieved chunks

This allows the system to answer questions about documents that weren't in the model's training data.
    """
    
    test_doc_path.write_text(test_content)
    print(f"✅ Created test document: {test_doc_path}")
    
    # Step 3: Test document processing
    print("\n3. Processing document...")
    try:
        chunks = rag_system.document_processor.process_document(str(test_doc_path))
        print(f"✅ Document processed into {len(chunks)} chunks")
        for i, chunk in enumerate(chunks[:2]):  # Show first 2 chunks
            print(f"   Chunk {i+1}: {chunk.content[:100]}...")
    except Exception as e:
        print(f"❌ Error processing document: {e}")
        return False
    
    # Step 4: Test embedding generation
    print("\n4. Generating embeddings...")
    try:
        embedded_chunks = rag_system.embedding_generator.generate_embeddings(chunks)
        print(f"✅ Generated embeddings for {len(embedded_chunks)} chunks")
        print(f"   Embedding dimension: {len(embedded_chunks[0].embedding)}")
    except Exception as e:
        print(f"❌ Error generating embeddings: {e}")
        return False
    
    # Step 5: Test vector database insertion
    print("\n5. Inserting into vector database...")
    try:
        inserted_ids = rag_system.vector_db.insert_embeddings(embedded_chunks)
        print(f"✅ Inserted {len(inserted_ids)} embeddings into vector DB")
    except Exception as e:
        print(f"❌ Error inserting into vector DB: {e}")
        return False
    
    # Step 6: Test querying
    print("\n6. Testing query...")
    try:
        result = rag_system.generate_response(
            query="What document types are supported by the RAG system?",
            max_chunks=5
        )
        print(f"✅ Query successful!")
        print(f"\n📝 Response:\n{result.response}\n")
        print(f"📚 Sources used: {result.retrieval_count}")
        if result.sources_used:
            print("\n📖 Citations:")
            print(result.get_citation_summary())
    except Exception as e:
        print(f"❌ Error querying: {e}")
        return False
    
    # Cleanup
    print("\n7. Cleaning up...")
    if test_doc_path.exists():
        test_doc_path.unlink()
        print("✅ Test document removed")
    
    print("\n" + "=" * 60)
    print("✅ All tests passed!")
    print("=" * 60)
    return True

if __name__ == "__main__":
    try:
        success = test_document_indexing()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
