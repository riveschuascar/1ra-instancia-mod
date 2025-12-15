import sqlite3
import pandas as pd

def get_tables_names():
    # Connect to the SQLite database
    conn = sqlite3.connect('database.sqlite')
    cursor = conn.cursor()

    # Execute a query to retrieve all table names
    cursor.execute('''SELECT name FROM sqlite_master WHERE type='table';''')
    tables = cursor.fetchall()

    conn.close()
    return tables

def print_tables_contents(tables):
    conn = sqlite3.connect('database.sqlite')

    # Load all tables into pandas DataFrames and print their contents
    for table_name in tables:
        table_name = table_name[0]
        df = pd.read_sql_query(f'SELECT * FROM {table_name} LIMIT 0;', conn)
        print(f"Table: {table_name}")
        print(df)
        print("\n")
    
    conn.close()