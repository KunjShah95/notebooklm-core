import logging
from typing import List, Dict, Any, Tuple, Optional, Union
from dataclasses import dataclass
import base64

from litellm import completion
from src.vector_database.milvus_vector_db import MilvusVectorDB
from src.embeddings.embedding_generator import EmbeddingGenerator

@dataclass
class MultimodalContent:
    """Represents multimodal content (text, images, etc.)"""
    type: str  # "text", "image", "audio", etc.
    content: Union[str, bytes]  # text content or binary data
    mime_type: Optional[str] = None  # e.g., "image/jpeg", "image/png"
    metadata: Optional[Dict[str, Any]] = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RAGResult:
    """Represents the result of RAG generation with citations"""
    query: str
    response: str
    sources_used: List[Dict[str, Any]]
    retrieval_count: int
    generation_tokens: Optional[int] = None
    multimodal_content: Optional[List[MultimodalContent]] = None
    
    def get_citation_summary(self) -> str:
        if not self.sources_used:
            return "No sources cited"
        
        source_summary = []
        for source in self.sources_used:
            source_info = f"• {source.get('source_file', 'Unknown')} ({source.get('source_type', 'unknown')})"
            if source.get('page_number'):
                source_info += f" - Page {source['page_number']}"
            source_summary.append(source_info)
        
        return "\n".join(source_summary)


class RAGGenerator:
    def __init__(
        self,
        embedding_generator: EmbeddingGenerator,
        vector_db: MilvusVectorDB,
        provider: str = "groq",
        model_name: str = "llama3-8b-8192"
    ):
        self.embedding_generator = embedding_generator
        self.vector_db = vector_db
        self.provider = provider.lower()
        self.model_name = model_name
        
        # Set model based on provider
        if self.provider == "groq":
            self.full_model_name = f"groq/{self.model_name}"
        elif self.provider == "ollama":
            self.full_model_name = f"ollama/{self.model_name}"
        else:
            self.full_model_name = self.model_name  # fallback
        
        self.model_name = model_name
        logger.info(f"RAG Generator initialized with {model_name}")
    
    def generate_response(
        self,
        query: str,
        max_chunks: int = 8,
        max_context_chars: int = 4000,
        top_k: int = 10,
        multimodal_content: Optional[List[MultimodalContent]] = None,
    ) -> RAGResult:

        if not query.strip():
            return RAGResult(
                query=query,
                response="Please provide a valid question.",
                sources_used=[],
                retrieval_count=0
            )
        
        try:
            logger.info(f"Generating response for: '{query[:50]}...'")
            
            # Step 1: Retrieve relevant chunks
            query_vector = self.embedding_generator.generate_query_embedding(query)
            search_results = self.vector_db.search(
                query_vector=query_vector.tolist(),
                limit=top_k
            )
            
            if not search_results:
                return RAGResult(
                    query=query,
                    response="I couldn't find any relevant information in the available documents to answer your question.",
                    sources_used=[],
                    retrieval_count=0
                )
            
            # Step 2: Format context with citations
            context, sources_info = self._format_context_with_citations(
                search_results, max_chunks, max_context_chars
            )
            
            # Step 3: Create citation-aware prompt
            prompt = self._create_rag_prompt(query, context)
            
            # Step 4: Generate response
            response = completion(model=self.full_model_name, messages=[{"role": "user", "content": prompt}], max_tokens=2000)["choices"][0]["message"]["content"]
            
            # Step 5: Create result object
            rag_result = RAGResult(
                query=query,
                response=response,
                sources_used=sources_info,
                retrieval_count=len(search_results),
                multimodal_content=multimodal_content
            )
            
            logger.info(f"Response generated successfully using {len(sources_info)} sources")
            return rag_result
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return RAGResult(
                query=query,
                response=f"I encountered an error while processing your question: {str(e)}",
                sources_used=[],
                retrieval_count=0,
                multimodal_content=multimodal_content
            )
    
    def _format_context_with_citations(
        self,
        search_results: List[Dict[str, Any]],
        max_chunks: int,
        max_context_chars: int
    ) -> Tuple[str, List[Dict[str, Any]]]:

        context_parts = []
        sources_info = []
        total_chars = 0
        for i, result in enumerate(search_results[:max_chunks]):
            citation_info = result['citation']
            source_file = citation_info.get('source_file', 'Unknown Source')
            source_type = citation_info.get('source_type', 'unknown')
            page_number = citation_info.get('page_number')
            
            citation_ref = f"[{i+1}]"
            chunk_content = result['content']
            chunk_text = f"{citation_ref} {chunk_content}"
            
            if total_chars + len(chunk_text) > max_context_chars and context_parts:
                break
            
            context_parts.append(chunk_text)
            total_chars += len(chunk_text)
            
            source_info = {
                'reference': citation_ref,
                'source_file': source_file,
                'source_type': source_type,
                'page_number': page_number,
                'chunk_id': result['id'],
                'relevance_score': result['score']
            }
            sources_info.append(source_info)
        
        formatted_context = '\n\n'.join(context_parts)

        return formatted_context, sources_info
    
    def _create_rag_prompt(self, query: str, context: str, multimodal_content: Optional[List[MultimodalContent]] = None) -> Union[str, List[Dict[str, Any]]]:
        if multimodal_content:
            # Create multimodal message format for vision-capable models
            content = []
            
            # Add text context and query
            text_prompt = f"""You are an AI assistant that answers questions based on provided source material and visual content. You must follow these citation rules:

CITATION REQUIREMENTS:
1. For each factual claim in your answer, include the citation reference number in square brackets [1], [2], etc.
2. Only use information from the provided context and images - do not add external knowledge
3. If you cannot find relevant information in the context or images, say so clearly
4. Be precise and accurate in your citations
5. When multiple sources support the same point, list all relevant citations like this [1], [2], [3].

CONTEXT (with citation references):
{context}

QUESTION: {query}

Please provide a comprehensive answer with proper citations. Make sure every factual statement is supported by a citation reference."""
            
            content.append({"type": "text", "text": text_prompt})
            
            # Add multimodal content (images, etc.)
            for item in multimodal_content:
                if item.type == "image":
                    # Convert image to base64 data URL
                    if isinstance(item.content, bytes):
                        base64_data = base64.b64encode(item.content).decode('utf-8')
                    else:
                        # Assume it's already base64 string
                        base64_data = item.content
                    
                    mime_type = item.mime_type or "image/jpeg"
                    data_url = f"data:{mime_type};base64,{base64_data}"
                    
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": data_url}
                    })
                elif item.type == "text":
                    # Additional text content
                    content.append({"type": "text", "text": str(item.content)})
                # Add other types as needed (audio, etc.)
            
            return content
        else:
            # Original text-only prompt
            prompt = f"""You are an AI assistant that answers questions based on provided source material. You must follow these citation rules:

CITATION REQUIREMENTS:
1. For each factual claim in your answer, include the citation reference number in square brackets [1], [2], etc.
2. Only use information from the provided context - do not add external knowledge
3. If you cannot find relevant information in the context, say so clearly
4. Be precise and accurate in your citations
5. When multiple sources support the same point, list all relevant citations like this [1], [2], [3].

CONTEXT (with citation references):
{context}

QUESTION: {query}

Please provide a comprehensive answer with proper citations. Make sure every factual statement is supported by a citation reference."""
            
            return prompt
    
    def generate_summary(
        self,
        max_chunks: int = 15,
        summary_length: str = "medium"
    ) -> RAGResult:
        try:
            summary_query = "main topics key findings important information overview"
            query_vector = self.embedding_generator.generate_query_embedding(summary_query)
            search_results = self.vector_db.search(
                query_vector=query_vector.tolist(),
                limit=max_chunks
            )
            
            if not search_results:
                return RAGResult(
                    query="Document Summary",
                    response="No documents available for summarization.",
                    sources_used=[],
                    retrieval_count=0
                )
            
            context, sources_info = self._format_context_with_citations(
                search_results, max_chunks, 6000
            )
            
            length_instructions = {
                'short': "Provide a concise 2-3 paragraph summary highlighting the most important points.",
                'medium': "Provide a comprehensive 4-5 paragraph summary covering key topics and findings.",
                'long': "Provide a detailed summary with multiple sections covering all major topics and supporting details."
            }
            
            summary_prompt = f"""You are tasked with creating a summary of the provided document content. Follow these guidelines:

1. {length_instructions.get(summary_length, length_instructions['medium'])}
2. Include citations [1], [2], etc. for all factual claims
3. Organize information logically with clear topics
4. Focus on the most important and relevant information
5. Maintain accuracy and cite sources properly

DOCUMENT CONTENT (with citation references):
{context}

Please provide a well-structured summary with proper citations:"""
            
            response = completion(model=self.full_model_name, messages=[{"role": "user", "content": summary_prompt}], max_tokens=1000)["choices"][0]["message"]["content"]
            
            return RAGResult(
                query="Document Summary",
                response=response,
                sources_used=sources_info,
                retrieval_count=len(search_results)
            )
            
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            return RAGResult(
                query="Document Summary",
                response=f"Error generating summary: {str(e)}",
                sources_used=[],
                retrieval_count=0
            )


if __name__ == "__main__":
    import os
    from src.document_preprocessing.doc_processor import DocumentProcessor
    from src.embeddings.embedding_generator import EmbeddingGenerator
    from src.vector_database.milvus_vector_db import MilvusVectorDB
    
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        print("Please set GROQ_API_KEY environment variable")
        exit(1)
    
    try:
        embedding_gen = EmbeddingGenerator()
        vector_db = MilvusVectorDB()
        rag_generator = RAGGenerator(
            embedding_generator=embedding_gen,
            vector_db=vector_db,
            provider="groq",
            model_name="llama3-8b-8192"
        )
        
        test_query = "What are the main findings discussed in the documents?"
        result = rag_generator.generate_response(test_query)
        
        print(f"Query: {result.query}")
        print(f"Response: {result.response}")
        print(f"\nSources Used ({len(result.sources_used)}):")
        print(result.get_citation_summary())
        
        summary_result = rag_generator.generate_summary(summary_length="medium")
        print(f"\nDocument Summary:")
        print(summary_result.response)
        
    except Exception as e:
        print(f"Error in RAG pipeline example: {e}")