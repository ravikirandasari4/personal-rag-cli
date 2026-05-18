import ollama
from typing import List, Dict

"""
LLMRetriever: A class to handle interactions with the LLM for generating responses based on retrieved context.
This class is responsible for constructing prompts based on retrieved document chunks and sending them to the LLM
to generate answers or summaries.
"""
class LLMRetriever:
    """LLMRetriever: A class to handle interactions with the LLM for generating responses based on retrieved context."""
    def __init__(self, model_name: str = "llama3.2:3b"):
        self.model_name = model_name
        self.client = ollama.Client()
    
    """Generate a response based on the query and retrieved context."""
    def generate_response(self, query: str, context_chunks: List[Dict]) -> str:
        """Generate response using retrieved context."""
        # Prepare context from retrieved chunks
        context = "\n\n".join([
            f"Document: {chunk['metadata']['filename']}\n{chunk['content']}"
            for chunk in context_chunks
        ])
        
        prompt = f"""Based on the following context documents, please answer the question.

Context:
{context}

Question: {query}

Answer: Provide a comprehensive answer based on the context above. If the information is not available in the context, please state that clearly."""

        try:
            response = self.client.chat(
                model=self.model_name,
                messages=[
                    {
                        'role': 'user',
                        'content': prompt
                    }
                ]
            )
            return response['message']['content']
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    """Generate a summary of the provided documents."""
    def summarize_documents(self, context_chunks: List[Dict]) -> str:
        """Generate a summary of the provided documents."""
        context = "\n\n".join([
            f"Document: {chunk['metadata']['filename']}\n{chunk['content']}"
            for chunk in context_chunks
        ])
        
        prompt = f"""Please provide a comprehensive summary of the following documents:

{context}

Summary: Create a detailed summary that captures the key points, main themes, and important information from these documents."""

        try:
            response = self.client.chat(
                model=self.model_name,
                messages=[
                    {
                        'role': 'user',
                        'content': prompt
                    }
                ]
            )
            return response['message']['content']
        except Exception as e:
            return f"Error generating summary: {str(e)}"