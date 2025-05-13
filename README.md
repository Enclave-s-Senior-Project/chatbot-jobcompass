# AI Chatbot with Website and Database Integration

This project implements an AI-powered chatbot that can answer questions based on both website content and database information. It uses LangChain, FastAPI, and PostgreSQL with vector storage for efficient semantic search.

## Features

-   Semantic search over website content using HuggingFace embeddings
-   Database querying capabilities
-   LLM powered responses
-   FastAPI REST API
-   PostgreSQL vector storage for efficient similarity search

## Prerequisites

-   Python 3.12+
-   PostgreSQL 12+ with pgvector extension
-   OpenAI API key
-   Node.js and npm (for website content)

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd chatbot
```

2. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Set up environment variables in `.env`:

```env
# OpenAI
OPENAI_API_KEY=your_openai_api_key

# Database
TYPEORM_CONNECTION=postgres
TYPEORM_HOST=localhost
TYPEORM_USERNAME=postgres
TYPEORM_PASSWORD=your_password
TYPEORM_DATABASE=your_database
TYPEORM_PORT=5432
TYPEORM_AUTOLOAD=true
TYPEORM_SYNCHRONIZE=true
TYPEORM_LOGGING=false
```

5. Set up PostgreSQL with pgvector:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

## Usage

1. Start the FastAPI server:

```bash
uvicorn main:app --reload
```

2. The API will be available at `http://localhost:8000`

3. API Endpoints:

    - POST `/chat`: Send chat messages
    - GET `/test`: Test endpoint

4. Example API call:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the main topic of the website?"}'
```

## Project Structure

```
chatbot/
├── main.py              # FastAPI application and chat endpoint
├── index.py            # Website content indexing
├── constants.py        # Configuration and constants
├── requirements.txt    # Python dependencies
└── .env               # Environment variables
```

## Dependencies

-   FastAPI: Web framework
-   LangChain: AI/ML framework
-   PostgreSQL: Database
-   pgvector: Vector similarity search
-   HuggingFace: Embeddings model
-   OpenAI: Gemini Flash 2.0

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
