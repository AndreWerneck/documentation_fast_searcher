from pydantic import BaseModel
import re
import uuid
from pathlib import Path
from typing import List, Optional
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings

class DocumentChunk(BaseModel):
    """ Represents a chunk of text extracted from a document.
    Each chunk has an ID, text content, raw text (optional), source file name, and metadata.
    """
    id : str
    text : str
    raw_text : Optional[str] = None
    source : str
    metadata : dict[str, str]
    

class MarkdownPreprocessor:
    """ Preprocesses markdown files in a given directory.
    It can load all markdown files, clean their content, and split them into chunks based on
    different chunking strategies (semantic, headers, or recursive).
    """
    def __init__(self, input_dir: str):
        self.input_dir = Path(input_dir)
        self.embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/multi-qa-mpnet-base-cos-v1")

    def load_all_markdown(self, chunking_type:str = 'semantic') -> List[DocumentChunk]:
        """ Loads all markdown files from the input directory and processes them into chunks.
        Args:
            chunking_type (str): The type of chunking to apply. Options are 'semantic', 'headers', or 'recursive'.
        Returns:
            List[DocumentChunk]: A list of DocumentChunk objects containing the processed text chunks.
        """
        chunks = []
        for md_file in self.input_dir.glob("*.md"):
            file_chunks = self._process_markdown_file(md_file, chunking_type=chunking_type)
            chunks.extend(file_chunks)
        return chunks

    def _clean_header(self, header: str) -> str:
        """ Cleans a markdown header by removing the leading hash symbols and whitespace.
        Args:
            header (str): The markdown header string to clean.
        Returns:
            str: The cleaned header text.
        """
        soup = BeautifulSoup(header, "html.parser")        
        return soup.get_text(strip=True).replace("#", "").strip()

    def _clean_text(self, text: str) -> str:
        """ Cleans the markdown text by removing HTML tags, code blocks, inline code, and normalizing whitespace.
        Args:
            text (str): The markdown text to clean.
        Returns:
            str: The cleaned text.
        """
        # Remove HTML tags
        soup = BeautifulSoup(text, "html.parser")
        text = soup.get_text()
        # Remove code blocks
        text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)

        # Remove inline code
        text = re.sub(r"`([^`]*)`", r"\1", text)

        # Normalize whitespace
        text = re.sub(r"\n{2,}", "\n", text)
        return text.strip()
    
    def _process_markdown_file(self, filepath: Path, chunking_type:str = 'semantic') -> List[DocumentChunk]:
        """ Processes a single markdown file, cleaning its content and splitting it into chunks.
        Args:
            filepath (Path): The path to the markdown file to process.
            chunking_type (str): The type of chunking to apply. Options are 'semantic', 'headers', or 'recursive'.
        Returns:
            List[DocumentChunk]: A list of DocumentChunk objects containing the processed text chunks.
        """
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        
        cleaned_content = self._clean_text(content)

        if chunking_type == 'headers':
            text_splitter = MarkdownHeaderTextSplitter(
                headers_to_split_on=[
                    # ("#", "title"),
                    ("##", "section"),
                ]
            )
            raw_docs = text_splitter.split_text(cleaned_content)
        
        elif chunking_type == 'recursive':
            text_splitter = RecursiveCharacterTextSplitter(
                separators=[". ", "! ", "? ", ".\n","?\n","!\n"],
                chunk_size=2048,
                chunk_overlap=128,
            )
            raw_docs = text_splitter.create_documents([cleaned_content])
        else: # semantic
            text_splitter = SemanticChunker(embeddings=self.embedding_model)

            raw_docs = text_splitter.create_documents([cleaned_content])

        chunks = []
        for doc in raw_docs:
            # Update header path
            metadata = doc.metadata
            text = doc.page_content
            chunk = DocumentChunk(
                id=str(uuid.uuid4()),
                text=text,
                source=filepath.name,
                metadata=metadata.copy(),
            )
            chunks.append(chunk)

        return chunks

if __name__ == "__main__":
    preprocessor = MarkdownPreprocessor("../sagemaker_documentation")
    chunks = preprocessor.load_all_markdown()

    print(f"Total chunks: {len(chunks)}")

    # Optional: Save to disk
    import json
    with open("../data/chunks.json", "w", encoding="utf-8") as f:
        json.dump([chunk.model_dump(mode='json') for chunk in chunks], f, indent=2, ensure_ascii=False)