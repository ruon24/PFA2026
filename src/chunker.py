"""
Text Chunker - Split text into overlapping chunks
"""

import tiktoken
from typing import List


class TextChunker:
    """Split text into overlapping chunks using tokenization"""
    
    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        """
        Initialize the chunker
        
        Args:
            chunk_size: Maximum tokens per chunk
            overlap: Number of overlapping tokens between chunks
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.enc = tiktoken.get_encoding("cl100k_base")
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of text chunks
        """
        tokens = self.enc.encode(text)
        chunks = []
        
        # Calculate step size (chunk_size - overlap)
        step = max(1, self.chunk_size - self.overlap)
        
        for i in range(0, len(tokens), step):
            chunk_tokens = tokens[i:i + self.chunk_size]
            chunk_text = self.enc.decode(chunk_tokens)
            
            if chunk_text.strip():
                chunks.append(chunk_text)
        
        return chunks
    
    def chunk_text_by_sentences(self, text: str, max_chunk_size: int = 500) -> List[str]:
        """
        Split text into chunks by sentences, respecting chunk size
        
        Args:
            text: Input text
            max_chunk_size: Maximum tokens per chunk
            
        Returns:
            List of text chunks
        """
        sentences = text.split('. ')
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip() + "." if sentence else ""
            test_chunk = current_chunk + " " + sentence if current_chunk else sentence
            
            if len(self.enc.encode(test_chunk)) <= max_chunk_size:
                current_chunk = test_chunk
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def get_token_count(self, text: str) -> int:
        """Get the number of tokens in text"""
        return len(self.enc.encode(text))