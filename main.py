import aiofiles
import aiofiles.os
import asyncio
import os
import re
import shutil
import json
import uuid

from crawl4ai import AsyncWebCrawler
from datetime import datetime
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain.document_loaders import UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain import OpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains.summarize import load_summarize_chain
from typing import Optional, Dict, List
from urllib.parse import unquote
from dataclasses import dataclass
import sqlite3
from datetime import datetime
import json
from langchain import PromptTemplate

# Load .env configuration
load_dotenv()

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Store crawl jobs status
crawl_jobs: Dict[str, dict] = {}

OUTPUT_DIR = os.getenv("OUTPUT_DIR", "output")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Add OPENAI_API_KEY to configuration
config = {}
config['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY', 'default_value')
config['OUTPUT_DIR'] = os.getenv("OUTPUT_DIR", "output")

@dataclass
class ProcessedDocument:
    title: str
    summary: str
    keywords: List[str]
    created_at: str
    original_path: str


def init_database():
    """Initialize SQLite database with required schema"""
    conn = sqlite3.connect('research_documents.db')
    cursor = conn.cursor()
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS documents (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        summary TEXT NOT NULL,
        keywords TEXT NOT NULL,
        created_at TIMESTAMP NOT NULL,
        original_path TEXT NOT NULL
    )
    ''')
    
    conn.commit()
    conn.close()


def save_to_database(doc: ProcessedDocument):
    """Save processed document to SQLite database"""
    conn = sqlite3.connect('research_documents.db')
    cursor = conn.cursor()
    
    cursor.execute('''
    INSERT INTO documents (title, summary, keywords, created_at, original_path)
    VALUES (?, ?, ?, ?, ?)
    ''', (
        doc.title,
        doc.summary,
        json.dumps(doc.keywords),
        doc.created_at,
        doc.original_path
    ))
    
    conn.commit()
    conn.close()


def generate_title(llm: OpenAI, text: str) -> str:
    """Generate a one-line title for the document"""
    title_template = """Please generate a clear, concise one-line title for the following text. 
    The title should capture the main topic and be under 10 words.
    
    Text: {text}
    
    Title:"""
    
    title_prompt = PromptTemplate(template=title_template, input_variables=["text"])
    
    return llm(title_prompt.format(text=text[:1000])).strip()


def extract_keywords(llm: OpenAI, summary: str) -> List[str]:
    """Extract key topics and keywords from the summary"""
    keyword_template = """Please extract 5-7 relevant keywords or key phrases from the following text.
    Return them as a comma-separated list with no explanations or additional text.
    
    Text: {text}
    
    Keywords:"""
    
    keyword_prompt = PromptTemplate(template=keyword_template, input_variables=["text"])
    
    keywords_text = llm(keyword_prompt.format(text=summary)).strip()
    return [k.strip() for k in keywords_text.split(',')]


def process_markdown_file(file_path: str, llm: OpenAI) -> ProcessedDocument:
    """Process a single markdown file and return structured results"""
    # Load and split the document
    loader = UnstructuredMarkdownLoader(file_path)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(documents)
    
    # Create embeddings and index
    embeddings = OpenAIEmbeddings()
    faiss_index = FAISS.from_documents(splits, embeddings)
    
    # Generate summary
    chain = load_summarize_chain(
        llm,
        chain_type="map_reduce",
        return_intermediate_steps=False,
        map_prompt=PromptTemplate(
            template="Summarize the following text:\n\n{text}\n\nSummary:",
            input_variables=["text"]
        ),
        combine_prompt=PromptTemplate(
            template="Combine these summaries into a coherent summary of the entire text:\n\n{text}\n\nFinal summary:",
            input_variables=["text"]
        )
    )
    summary = chain.run(splits)
    
    # Generate title and keywords
    title = generate_title(llm, summary)
    keywords = extract_keywords(llm, summary)
    
    return ProcessedDocument(
        title=title,
        summary=summary,
        keywords=keywords,
        created_at=datetime.now().isoformat(),
        original_path=file_path
    )


def summarize_markdown(research_dir: str):
    """Process all markdown files in the research directory"""
    # Initialize database
    init_database()
    
    # Initialize OpenAI model
    llm = OpenAI(temperature=0)
    
    # Process all markdown files
    for root, _, files in os.walk(research_dir):
        for file in files:
            if file.endswith('.md'):
                file_path = os.path.join(root, file)
                try:
                    # Process the markdown file
                    processed_doc = process_markdown_file(file_path, llm)
                    
                    # Save to database
                    save_to_database(processed_doc)
                    
                    # Save to JSON for backup
                    output_json = {
                        'title': processed_doc.title,
                        'summary': processed_doc.summary,
                        'keywords': processed_doc.keywords,
                        'created_at': processed_doc.created_at,
                        'original_path': processed_doc.original_path
                    }
                    
                    json_path = f"{os.path.splitext(file_path)[0]}_processed.json"
                    with open(json_path, 'w') as f:
                        json.dump(output_json, f, ensure_ascii=False, indent=2)
                        
                except Exception as e:
                    print(f"Error processing {file_path}: {str(e)}")


class CrawlRequest(BaseModel):
    url: str
    limit: int = 10


class CrawlResponse(BaseModel):
    job_id: str
    status: str
    progress: int = 0
    total_pages: int = 0
    current_url: Optional[str] = None


def clean_path(url: str, base_url: str) -> str:
    """Extract and clean the path from URL relative to base URL"""
    url = unquote(url)
    base_url = unquote(base_url)
    path = url.replace(base_url, "").lstrip("/")
    if "#" in path:
        path = path.split("#")[0]
    path = re.sub(r"[^\w\s-]", "", path)
    return re.sub(r"\s+", "_", path.strip()).lower()


async def process_url(url: str, output_dir: str, crawler: AsyncWebCrawler, job_id: str):
    """Process a single URL and save markdown"""
    try:
        result = await crawler.arun(
            url=url, remove_overlay_elements=True, bypass_cache=True
        )
        if result.success:
            metadata = result.metadata
            title = metadata["title"]
            clean_title = re.sub(r"[^\w\s-]", "", title)
            clean_title = re.sub(r"\s+", "_", clean_title.strip())
            path_suffix = clean_path(url, crawl_jobs[job_id]["base_url"])
            filename = (
                f"{clean_title.lower()}_{path_suffix}.md"
                if path_suffix
                else f"{clean_title.lower()}.md"
            )
            filepath = os.path.join(output_dir, filename)
            async with aiofiles.open(filepath, "w") as f:
                await f.write(result.markdown)
            return result.links.get("internal", [])
    except Exception as e:
        print(f"Error processing {url}: {str(e)}")
    return []


async def crawl_website(job_id: str, url: str, limit: int):
    """Recursively crawl website and update job status"""
    try:
        output_dir = os.path.join(config['OUTPUT_DIR'], f"output_{job_id}")
        os.makedirs(output_dir, exist_ok=True)
        crawl_jobs[job_id]["base_url"] = url
        async with AsyncWebCrawler(verbose=True) as crawler:
            processed_urls = set()
            urls_to_process = {url}
            while urls_to_process and len(processed_urls) < limit:
                current_url = urls_to_process.pop()
                if current_url in processed_urls:
                    continue
                crawl_jobs[job_id].update(
                    {
                        "status": "processing",
                        "progress": len(processed_urls),
                        "current_url": current_url,
                    }
                )
                internal_links = await process_url(
                    current_url, output_dir, crawler, job_id
                )
                processed_urls.add(current_url)
                for link in internal_links:
                    link_url = link.get("href", "") if isinstance(link, dict) else link
                    if (
                        link_url
                        and link_url.startswith(url)
                        and link_url not in processed_urls
                    ):
                        urls_to_process.add(link_url)
        
        # Run summarization and processing after crawl is complete
        try:
            summarize_markdown(output_dir)
        except Exception as e:
            print(f"Error during summarization: {str(e)}")
        
        crawl_jobs[job_id].update(
            {
                "status": "completed",
                "progress": len(processed_urls),
                "total_pages": len(processed_urls),
            }
        )
        
    except Exception as e:
        crawl_jobs[job_id]["status"] = "failed"
        print(f"Crawl failed: {str(e)}")


@app.post("/api/crawl", response_model=CrawlResponse)
async def start_crawl(request: CrawlRequest):
    job_id = str(uuid.uuid4())
    crawl_jobs[job_id] = {
        "status": "starting",
        "progress": 0,
        "total_pages": 0,
        "base_url": request.url,
    }
    asyncio.create_task(crawl_website(job_id, request.url, request.limit))
    return CrawlResponse(job_id=job_id, status="starting")


@app.get("/api/status/{job_id}", response_model=CrawlResponse)
async def get_status(job_id: str):
    if job_id not in crawl_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    job = crawl_jobs[job_id]
    return CrawlResponse(job_id=job_id, **job)


@app.get("/api/download/{job_id}")
async def download_results(job_id: str):
    if job_id not in crawl_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = crawl_jobs[job_id]
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job not completed")

    zip_path = f"output/output_{job_id}.zip"
    if not os.path.exists(zip_path):
        raise HTTPException(status_code=404, detail="Results not found")

    filename = "crawl_results.zip"
    return FileResponse(zip_path, media_type="application/zip", filename=filename)


@app.get("/")
async def read_root():
    try:
        return FileResponse("static/index.html")
    except Exception as e:
        return {"status": "ok"}


# Additional endpoints for accessing processed data

@app.get("/api/documents")
async def get_documents():
    """Retrieve all processed documents from the database"""
    conn = sqlite3.connect('research_documents.db')
    cursor = conn.cursor()
    
    cursor.execute('SELECT * FROM documents ORDER BY created_at DESC')
    rows = cursor.fetchall()
    
    documents = []
    for row in rows:
        documents.append({
            'id': row[0],
            'title': row[1],
            'summary': row[2],
            'keywords': json.loads(row[3]),
            'created_at': row[4],
            'original_path': row[5]
        })
    
    conn.close()
    return {'documents': documents}


@app.get("/api/documents/{doc_id}")
async def get_document(doc_id: int):
    """Retrieve a specific document by ID"""
    conn = sqlite3.connect('research_documents.db')
    cursor = conn.cursor()
    
    cursor.execute('SELECT * FROM documents WHERE id = ?', (doc_id,))
    row = cursor.fetchone()
    
    if not row:
        raise HTTPException(status_code=404, detail="Document not found")
    
    document = {
        'id': row[0],
        'title': row[1],
        'summary': row[2],
        'keywords': json.loads(row[3]),
        'created_at': row[4],
        'original_path': row[5]
    }
    
    conn.close()
    return document


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)