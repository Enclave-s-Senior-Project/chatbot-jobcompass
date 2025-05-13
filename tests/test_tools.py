# test_tools.py
from app.tools import website_tool, db_tool

print(website_tool.func("What’s on the about page?"))
print(db_tool.func("SELECT price FROM products WHERE name = 'Product X'"))
