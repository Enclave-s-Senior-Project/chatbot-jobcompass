from langchain.agents import Tool
from langchain_community.utilities import SQLDatabase
from constants import main_database_url
import json


# Database tool
db = SQLDatabase.from_uri(database_uri=main_database_url)


def database_query(query):
    try:
        if not query.strip().lower().startswith("select"):
            return "Only SELECT queries are allowed."
        result = db.run(query)
        return json.dumps(result) if result else "No results found."
    except Exception as e:
        return f"Error executing query: {str(e)}"


db_tool = Tool(
    name="Database",
    func=database_query,
    description="Run SELECT SQL queries on the PostgreSQL database. Example: SELECT * FROM products WHERE name = 'Product X'.",
)
