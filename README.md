# AI Assistant Desktop App

A desktop application built with Electron and Python that serves as an AI assistant, inspired by and built upon the work from [crawl4ai-frontend](https://github.com/f4ww4z/crawl4ai-frontend) and [blog_gpt](https://github.com/yaohui-wyh/blog_gpt).

## Prerequisites

Before you begin, ensure you have the following installed:
- [Node.js](https://nodejs.org/) (v14 or higher)
- [Python](https://www.python.org/) (v3.8 or higher)
- [Docker](https://www.docker.com/) (optional, for containerized deployment)
- [Poetry](https://python-poetry.org/) (Python dependency management)

## Installation

### Option 1: Local Development Setup

1. Clone the repository:
```bash
git clone <your-repository-url>
cd <repository-name>
```

2. Install Node.js dependencies:
```bash
npm install
```

3. Install Python dependencies using Poetry:
```bash
poetry install
```

4. Create and activate the Python virtual environment:
```bash
poetry shell
```

### Option 2: Docker Setup

1. Build and run using Docker Compose:
```bash
docker-compose up --build
```

## Running the Application

### Local Development
1. Start the application:
```bash
npx electron .
```

The application window should open automatically.

### Docker Environment
The application will be accessible through the container as specified in the Docker configuration.

## Project Structure

- `main.js` - Main Electron process file
- `preload.js` - Preload script for Electron
- `main.py` - Python backend server
- `static/` - Static assets directory
- `package.json` - Node.js dependencies and scripts
- `pyproject.toml` - Python project configuration
- `poetry.lock` - Python dependency lock file
- `Dockerfile` - Docker configuration
- `docker-compose.yml` - Docker Compose configuration

## Acknowledgments

This project is built upon the following open-source projects:
- [crawl4ai-frontend](https://github.com/f4ww4z/crawl4ai-frontend)
- [blog_gpt](https://github.com/yaohui-wyh/blog_gpt)

## License

This project is licensed under the terms included in the LICENSE file.
