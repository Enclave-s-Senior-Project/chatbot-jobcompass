import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import psycopg2
import enterprise_embed
import job_embed
import website_content_embed


conn = psycopg2.connect(
    dbname=os.getenv("VECTOR_DB_DATABASE"),
    user=os.getenv("VECTOR_DB_USERNAME"),
    password=os.getenv("VECTOR_DB_PASSWORD"),
    host=os.getenv("VECTOR_DB_HOST"),
    port=os.getenv("VECTOR_DB_PORT"),
)
cursor = conn.cursor()

print("Starting embedding process...")
print("Clearing existing embeddings...")
print("==========================")

cursor.execute("delete from langchain_pg_embedding;")
rowcount = cursor.rowcount
if rowcount == 0:
    print("No existing embeddings found.")
else:
    print(f"Cleared {rowcount} existing embeddings.")

conn.commit()
print("Existing embeddings cleared successfully.")
print("==========================")

try:
    # Call the actual functions from the modules
    print("\n1. Starting website content embedding...")
    website_content_embed.embed_website_content()
    print("✓ Website content embedding completed.")
    print("==========================")

    print("\n2. Starting job embedding...")
    job_embed.main()
    print("✓ Job embedding completed.")
    print("==========================")

    print("\n3. Starting enterprise embedding...")
    enterprise_embed.main()
    print("✓ Enterprise embedding completed.")
    print("==========================")

    print("\n🎉 All embeddings completed successfully!")

except Exception as e:
    print(f"❌ Error during embedding process: {str(e)}")
    raise
finally:
    conn.close()
    print("==========================")
    print("Database connection closed.")
