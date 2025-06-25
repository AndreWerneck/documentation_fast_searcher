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
    id : str
    text : str
    raw_text : Optional[str] = None
    source : str
    metadata : dict[str, str]
    

class MarkdownPreprocessor:
    def __init__(self, input_dir: str):
        self.input_dir = Path(input_dir)
        self.embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/multi-qa-mpnet-base-cos-v1")

    def load_all_markdown(self, chunking_type:str = 'semantic') -> List[DocumentChunk]:
        chunks = []
        for md_file in self.input_dir.glob("*.md"):
            file_chunks = self._process_markdown_file(md_file, chunking_type=chunking_type)
            chunks.extend(file_chunks)
        return chunks

    def _clean_header(self, header: str) -> str:
        soup = BeautifulSoup(header, "html.parser")        
        return soup.get_text(strip=True).replace("#", "").strip()

    def _clean_text(self, text: str) -> str:
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

    # def _split_markdown_by_headers(self, content: str) -> List[tuple[str, str]]:
    #     # Regex to capture headers and the text that follows them
    #     pattern = r"(#{1,6} .+?)\n(?:(?!(?:#{1,6} )).*\n?)*"
    #     matches = re.finditer(pattern, content, re.MULTILINE)

    #     sections = []
    #     for match in matches:
    #         header_line = match.group(0).splitlines()[0]
    #         body = match.group(0)[len(header_line):].strip()
    #         sections.append((header_line, body))

    #     return sections

if __name__ == "__main__":
    preprocessor = MarkdownPreprocessor("sagemaker_documentation")
    chunks = preprocessor.load_all_markdown()

    print(f"Total chunks: {len(chunks)}")

    # Optional: Save to disk
    import json
    with open("data/chunks.json", "w", encoding="utf-8") as f:
        json.dump([chunk.model_dump(mode='json') for chunk in chunks], f, indent=2, ensure_ascii=False)