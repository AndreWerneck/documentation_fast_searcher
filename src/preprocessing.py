from pydantic import BaseModel
import re
import uuid
from pathlib import Path
from typing import List, Optional
from bs4 import BeautifulSoup
from langchain_text_splitters import MarkdownHeaderTextSplitter

class DocumentChunk(BaseModel):
    id : str
    text : str
    raw_text : Optional[str] = None
    source : str
    metadata : dict[str, str]
    

class MarkdownPreprocessor:
    def __init__(self, input_dir: str):
        self.input_dir = Path(input_dir)

    def load_all_markdown(self) -> List[DocumentChunk]:
        chunks = []
        for md_file in self.input_dir.glob("*.md"):
            file_chunks = self._process_markdown_file(md_file)
            chunks.extend(file_chunks)
        return chunks

    def _clean_header(self, header: str) -> str:
        # Try to extract name attribute from <a> tag
        soup = BeautifulSoup(header, "html.parser")
        anchor = soup.find("a")

        if anchor and anchor.has_attr("name"):
            return anchor["name"]
        
        # Fallback to header text if no <a name=...>
        return soup.get_text(strip=True).replace("#", "").strip()


    def _clean_chunk_text(self, text: str) -> str:
    # Remove code blocks
        text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)

        # Remove inline code
        text = re.sub(r"`([^`]*)`", r"\1", text)

        # Normalize whitespace
        text = re.sub(r"\n{2,}", "\n", text)
        text = text.strip()

        return text
    
    def _process_markdown_file(self, filepath: Path) -> List[DocumentChunk]:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        
        splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[
            ("#", "title"),
            ("##", "section"),
            ("###", "subsection"),
            ("####", "subsubsection")
        ])
        docs = splitter.split_text(content)

        chunks = []

        for doc in docs:
            # Update header path
            metadata = doc.metadata
            raw_text = doc.page_content
            clean_text = self._clean_chunk_text(raw_text)
            if not clean_text:
                continue

            for level, text in metadata.items():
                cleaned_header = self._clean_header(text)
                if cleaned_header:
                    metadata[level] = cleaned_header

            chunk = DocumentChunk(
                id=str(uuid.uuid4()),
                text=clean_text,
                raw_text=raw_text.strip(),
                source=filepath.name,
                metadata=metadata.copy(),
            )
            chunks.append(chunk)

        return chunks

    def _split_markdown_by_headers(self, content: str) -> List[tuple[str, str]]:
        # Regex to capture headers and the text that follows them
        pattern = r"(#{1,6} .+?)\n(?:(?!(?:#{1,6} )).*\n?)*"
        matches = re.finditer(pattern, content, re.MULTILINE)

        sections = []
        for match in matches:
            header_line = match.group(0).splitlines()[0]
            body = match.group(0)[len(header_line):].strip()
            sections.append((header_line, body))

        return sections

if __name__ == "__main__":
    preprocessor = MarkdownPreprocessor("sagemaker_documentation")
    chunks = preprocessor.load_all_markdown()

    print(f"Total chunks: {len(chunks)}")

    # Optional: Save to disk
    import json
    with open("data/chunks.json", "w", encoding="utf-8") as f:
        json.dump([chunk.model_dump(mode='json') for chunk in chunks], f, indent=2, ensure_ascii=False)