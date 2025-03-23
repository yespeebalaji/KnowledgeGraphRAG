import PyPDF2
import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Union, Generator
from pathlib import Path
import io
import concurrent.futures
import functools
import gc
import logging
from tqdm import tqdm
import os
import sys
import importlib.util

# Import local config module
def import_local_module(module_name: str):
    """Import a module from the local app directory."""
    module_path = os.path.join(os.path.dirname(__file__), f"{module_name}.py")
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# Import local configuration
config = import_local_module("config")
CHUNK_SIZE = config.CHUNK_SIZE
CHUNK_OVERLAP = config.CHUNK_OVERLAP
ALLOWED_FILE_TYPES = config.ALLOWED_FILE_TYPES
MAX_FILE_SIZE = config.MAX_FILE_SIZE

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Handles document processing from various sources (PDF, URL, text)."""
    
    def __init__(self):
        self.supported_types = ALLOWED_FILE_TYPES
        self._cache = {}
        self.BATCH_SIZE = 5  # Process 5 pages at a time for better memory management
    
    def _process_page(self, page) -> str:
        """Process a single PDF page with buffered text extraction."""
        page_num = page.page_number if hasattr(page, 'page_number') else -1
        if page_num in self._cache:
            return self._cache[page_num]

        # Use StringIO as a text buffer for memory efficiency
        buffer = io.StringIO()
        try:
            # Extract text directly to buffer
            text = page.extract_text()
            if text:
                buffer.write(text)
            
            result = buffer.getvalue()
            if page_num != -1:
                self._cache[page_num] = result
            return result
        finally:
            buffer.close()
    
    def _process_batch(self, pages: list, start_idx: int) -> List[str]:
        """Process a batch of pages."""
        results = []
        for i, page in enumerate(pages, start=start_idx):
            page.page_number = i  # Add page number for caching
            results.append(self._process_page(page))
        return results
    
    def _page_generator(self, pdf_reader) -> Generator[str, None, None]:
        """Generate processed pages in batches to manage memory."""
        total_pages = len(pdf_reader.pages)
        current_batch = []
        
        # Log processing start with summary
        logger.info("\n" + "="*80)
        logger.info("Starting PDF Processing")
        logger.info(f"Total Pages: {total_pages}")
        logger.info(f"Batch Size: {self.BATCH_SIZE}")
        logger.info(f"Estimated batches: {(total_pages + self.BATCH_SIZE - 1) // self.BATCH_SIZE}")
        logger.info("="*80 + "\n")
        
        progress_bar = tqdm(total=total_pages, desc="Processing PDF pages")
        
        try:
            for page_num in range(total_pages):
                try:
                    page = pdf_reader.pages[page_num]
                    page.page_number = page_num
                    current_batch.append(page)
                    
                    if len(current_batch) >= self.BATCH_SIZE:
                        texts = self._process_batch(current_batch, page_num - len(current_batch) + 1)
                        for idx, text in enumerate(texts):
                            page_number = page_num - len(texts) + idx + 1
                            logger.info("\n" + "="*80)
                            logger.info(f"Processing PDF - Page {page_number + 1}")
                            logger.info("-"*80)
                            content = text.strip() if text else "[Empty page]"
                            logger.info(f"Character count: {len(content)}")
                            logger.info(f"Word count: {len(content.split())}")
                            logger.info("-"*80)
                            logger.info("Content:")
                            logger.info(content)
                            logger.info("="*80 + "\n")
                        yield from texts
                        progress_bar.update(len(texts))
                        current_batch = []
                        gc.collect()
                except Exception as e:
                    logger.error(f"Error processing page {page_num}: {str(e)}")
                    yield f"[Error on page {page_num + 1}]"
                    continue
            
            # Process remaining pages
            if current_batch:
                try:
                    texts = self._process_batch(current_batch, total_pages - len(current_batch))
                    for idx, text in enumerate(texts):
                        page_number = total_pages - len(texts) + idx
                        logger.info("\n" + "="*80)
                        logger.info(f"Processing PDF - Page {page_number + 1}")
                        logger.info("-"*80)
                        content = text.strip() if text else "[Empty page]"
                        logger.info(f"Character count: {len(content)}")
                        logger.info(f"Word count: {len(content.split())}")
                        logger.info("-"*80)
                        logger.info("Content:")
                        logger.info(content)
                        logger.info("="*80 + "\n")
                    yield from texts
                    progress_bar.update(len(texts))
                    gc.collect()
                except Exception as e:
                    logger.error(f"Error processing final batch: {str(e)}")
                    for i in range(len(current_batch)):
                        yield f"[Error on page {total_pages - len(current_batch) + i + 1}]"
        
        finally:
            progress_bar.close()
            # Clear the cache to free memory
            self._cache.clear()
            gc.collect()
    
    def process_pdf(self, file_path: Union[str, Path, io.BytesIO]) -> str:
        """Extract text from PDF files using optimized batch processing."""
        try:
            if isinstance(file_path, (str, Path)):
                pdf_file = open(file_path, 'rb')
            else:
                pdf_file = file_path
            
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            # Process pages in batches and join with newlines
            processed_text = '\n'.join(self._page_generator(pdf_reader))
            
            if isinstance(file_path, (str, Path)):
                pdf_file.close()
            
            return processed_text.strip()
        except Exception as e:
            raise Exception(f"Error processing PDF: {str(e)}")
    
    def process_url(self, url: str) -> str:
        """Scrape and extract text content from a URL."""
        try:
            response = requests.get(url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Extract text content
            text = soup.get_text()
            
            # Clean and normalize text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            return text
        except Exception as e:
            raise Exception(f"Error processing URL: {str(e)}")
    
    def chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks for processing using a memory-efficient approach."""
        def generate_chunks():
            """Generator function to yield chunks one at a time."""
            start = 0
            text_length = len(text)
            
            while start < text_length:
                # Process text in smaller windows
                window_end = min(start + CHUNK_SIZE * 2, text_length)
                current_text = text[start:window_end]
                
                # Find the actual chunk end within the window
                chunk_end = min(CHUNK_SIZE, len(current_text))
                
                if chunk_end < len(current_text):
                    # Try to find natural breakpoints
                    last_period = current_text.rfind('.', 0, chunk_end)
                    last_space = current_text.rfind(' ', 0, chunk_end)
                    
                    if last_period > chunk_end // 2:
                        chunk_end = last_period + 1
                    elif last_space > chunk_end // 2:
                        chunk_end = last_space
                
                chunk = current_text[:chunk_end].strip()
                if chunk:
                    yield chunk
                
                # Move the start position for the next chunk
                start += max(chunk_end - CHUNK_OVERLAP, 1)
                
                # Free up memory
                del current_text
                gc.collect()
        
        # Convert generator to list, processing one chunk at a time
        return list(generate_chunks())
    
    def extract_metadata(self, text: str) -> Dict:
        """Extract basic metadata from the document using memory-efficient processing."""
        def count_chunks():
            """Generator-based chunk counting."""
            count = 0
            for _ in self.chunk_text(text):
                count += 1
            return count
        
        return {
            'length': len(text),
            'chunk_count': count_chunks(),
            'estimated_tokens': sum(1 for _ in text.split())
        }
    
    def validate_file(self, file_name: str, file_size: int) -> bool:
        """Validate file type and size."""
        file_extension = file_name.split('.')[-1].lower()
        
        if file_extension not in self.supported_types:
            raise ValueError(f"Unsupported file type. Allowed types: {', '.join(self.supported_types)}")
        
        if file_size > MAX_FILE_SIZE * 1024 * 1024:  # Convert MB to bytes
            raise ValueError(f"File size exceeds maximum limit of {MAX_FILE_SIZE}MB")
        
        return True
    
    def process_document(self, source: Union[str, Path, io.BytesIO], source_type: str) -> Dict:
        """
        Process document from various sources and return text chunks and metadata.
        
        Args:
            source: File path, URL, or file object
            source_type: Type of source ('pdf', 'url', 'text')
            
        Returns:
            Dict containing processed chunks and metadata
        """
        try:
            if source_type == 'pdf':
                text = self.process_pdf(source)
            elif source_type == 'url':
                text = self.process_url(source)
            elif source_type == 'text':
                text = source if isinstance(source, str) else source.read()
            else:
                raise ValueError(f"Unsupported source type: {source_type}")
            
            chunks = self.chunk_text(text)
            metadata = self.extract_metadata(text)
            
            return {
                'chunks': chunks,
                'metadata': metadata,
                'original_text': text
            }
        except Exception as e:
            raise Exception(f"Error processing document: {str(e)}")
