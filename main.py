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
from typing import Optional, Dict
from urllib.parse import unquote


# Load .env configuration
load_dotenv()

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Store crawl jobs status
crawl_jobs: Dict[str, dict] = {}

OUTPUT_DIR = os.getenv("OUTPUT_DIR", "output")


class CrawlRequest(BaseModel):
    url: str
    limit: int = 10


class CrawlResponse(BaseModel):
    job_id: str
    status: str
    progress: int = 0
    total_pages: int = 0
    current_url: Optional[str] = None


class DownloadRequest(BaseModel):
    download_path: str


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
        output_dir = os.path.join(OUTPUT_DIR, f"output_{job_id}")
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
        
        # FIXME: Add summarization and keyword extraction
        shutil.rmtree(output_dir)
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


async def unzip_and_cleanup(zip_path: str, extract_dir: str):
    try:
        # Ensure the extraction directory exists
        await aiofiles.os.makedirs(extract_dir, exist_ok=True)

        # Copy the zip file to the target directory
        zip_filename = os.path.basename(zip_path)
        target_zip_path = os.path.join(extract_dir, zip_filename)
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, shutil.copy2, zip_path, target_zip_path)

        # Unzip the file in the new directory
        await loop.run_in_executor(
            None, shutil.unpack_archive, target_zip_path, extract_dir
        )

        # Remove the zip file from both the source and target locations
        await aiofiles.os.remove(zip_path)
        await aiofiles.os.remove(target_zip_path)

        return True
    except Exception as e:
        print(f"Error in unzip_and_cleanup: {str(e)}")
        return False


def summarize_markdown(files: list):
    """Summarize and extract keywords from markdown files"""
    for file_path in files:
        loader = UnstructuredMarkdownLoader(file_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000, chunk_overlap=200
        )
        splits = text_splitter.split_documents(documents)
        embeddings = OpenAIEmbeddings()
        faiss_index = FAISS.from_documents(splits, embeddings)
        retriever = faiss_index.as_retriever()
        llm = OpenAI(temperature=0)
        chain = load_summarize_chain(llm, chain_type="map_reduce")
        result = chain.run({"input_documents": splits})
        summary_path = f"{os.path.splitext(file_path)[0]}_summary.json"
        with open(summary_path, "w") as summary_file:
            json.dump({"summary": result}, summary_file, ensure_ascii=False, indent=4)


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


@app.get("/api/healthcheck")
async def healthcheck():
    return {"status": "ok"}


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

    # filename = datetime.strftime(datetime.now(), "%Y-%m-%d_%H-%M-%S") + "_crawl_results.zip"
    filename = "crawl_results.zip"

    return FileResponse(zip_path, media_type="application/zip", filename=filename)


@app.post("/api/process/{job_id}")
async def process_downloaded_file(job_id: str):
    """Process the downloaded file after user confirms download"""
    zip_path = f"{DOWNLOAD_DIR}/crawl_results.zip"
    if not os.path.exists(zip_path):
        raise HTTPException(status_code=404, detail="Zip file not found")

    try:
        await handle_post_download(job_id, zip_path)
        return {"status": "success", "message": "File processed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


DOWNLOAD_DIR = os.getenv("DOWNLOAD_DIR", "/app/Downloads")
RESEARCH_DIR = os.getenv("RESEARCH_DIR", "/app/Research")


async def handle_post_download(job_id: str, zip_path: str):
    """Handle post-download processing of the zip file"""
    try:
        print(f"Processing downloaded file for job {job_id}")
        # Create research directory if it doesn't exist
        os.makedirs(RESEARCH_DIR, exist_ok=True)

        # Generate timestamped folder name
        timestamp = datetime.strftime(datetime.now(), "%Y-%m-%d_%H-%M-%S")
        extract_dir = os.path.join(RESEARCH_DIR, f"crawl_results_{timestamp}")

        # Create directory for extracted files
        os.makedirs(extract_dir, exist_ok=True)

        success = await unzip_and_cleanup(zip_path, extract_dir)
        if not success:
            print(f"Failed to process downloaded file for job {job_id}")
    except Exception as e:
        print(f"Error in post-download processing for job {job_id}: {str(e)}")


# Serve index.html
@app.get("/")
async def read_root():
    try:
        return FileResponse("static/index.html")
    except Exception as e:
        # Return a simple JSON response if file not found
        return {"status": "ok"}
