# JobCompass AI Chatbot

A sophisticated AI-powered chatbot designed to assist users with job search and career guidance. The system combines website content analysis, database integration, and advanced AI capabilities to provide personalized career advice and job search assistance.

## Features

-   Semantic search over job listings and career resources using advanced embeddings
-   Real-time database querying for job market insights
-   LLM-powered responses with context-aware career guidance
-   FastAPI REST API with comprehensive documentation
-   PostgreSQL vector storage for efficient similarity search

## Prerequisites

-   Python 3.12+
-   PostgreSQL 12+ with pgvector extension
-   OpenAI API key

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/jobcompass.git
cd jobcompass
```

2. Set up the backend:

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt

# Download NLTK stopwords corpus (required for text processing)
python -c "import nltk; nltk.download('stopwords')"

# Set up environment variables in .env
cp .env.example .env
# Edit .env with your configuration
```

3. Set up PostgreSQL with pgvector:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

## Configuration

Create a `.env` file in the root directory with the following variables:

```env
# OpenAI
OPENAI_API_KEY=your_openai_api_key

# Database
VECTOR_DB_CONNECTION=postgres
VECTOR_DB_HOST=localhost
VECTOR_DB_USERNAME=postgres
VECTOR_DB_PASSWORD=your_password
VECTOR_DB_DATABASE=your_database
VECTOR_DB_PORT=5432
VECTOR_DB_AUTOLOAD=true
VECTOR_DB_SYNCHRONIZE=true
VECTOR_DB_LOGGING=false


```

## Usage

1. Start the backend server:

```bash
# From the root directory
uvicorn app.main:app --reload
```

2. Access the application:
    - API Documentation: http://localhost:8000/docs

## Project Structure

```
jobcompass/

├── main.py            # FastAPI application entry point
├── index.py           # Content indexing and processing
├── constants.py       # Configuration and constants
├── requirements.txt   # Python dependencies
└── .env              # Environment variables
```

## Dependencies

### Backend

-   FastAPI: Modern web framework for building APIs
-   LangChain: Framework for LLM applications
-   PostgreSQL: Advanced database system
-   pgvector: Vector similarity search extension
-   HuggingFace: Embeddings and models
-   OpenAI: Language model integration

### Frontend

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
