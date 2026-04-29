"""
Query Engine - Ollama LLM integration
"""

import ollama
from typing import Optional, Dict


class QueryEngine:
    """Query Ollama for answers based on retrieved context"""
    
    def __init__(self, model: str = "llama3.2"):
        """
        Initialize the query engine
        
        Args:
            model: Ollama model name
        """
        self.model = model
    
    def generate(
        self,
        prompt: str,
        context: str,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Generate a response using Ollama
        
        Args:
            prompt: User question/prompt
            context: Retrieved context from vector store
            system_prompt: Optional system prompt
            
        Returns:
            Generated response
        """
        if system_prompt is None:
            system_prompt = """You are a helpful AI assistant. 
Answer questions based only on the provided context.
If the context doesn't contain relevant information, say so."""
        
        full_prompt = f"""Context:
{context}

Question: {prompt}

Answer:"""
        
        response = ollama.generate(
            model=self.model,
            prompt=full_prompt,
            system=system_prompt,
            options={
                "temperature": 0.3,
                "top_p": 0.9,
                "num_ctx": 4096
            }
        )
        
        return response['response']
    
    def chat(self, prompt: str, context: str) -> str:
        """
        Chat with Ollama using context
        
        Args:
            prompt: User message
            context: Retrieved context
            
        Returns:
            Chat response
        """
        full_prompt = f"""Based on the following context, answer the question.
If the context doesn't contain relevant information, say so.

Context:
{context}

Question: {prompt}

Answer:"""
        
        response = ollama.chat(
            model=self.model,
            messages=[
                {"role": "user", "content": full_prompt}
            ]
        )
        
        return response['message']['content']
    
    def list_models(self) -> Dict:
        """
        List available Ollama models
        
        Returns:
            Dictionary of available models
        """
        return ollama.list()
    
    def check_connection(self) -> bool:
        """
        Check if Ollama is running and accessible
        
        Returns:
            True if connected, False otherwise
        """
        try:
            ollama.list()
            return True
        except Exception:
            return False