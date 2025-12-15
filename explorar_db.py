import sqlite3
import pandas as pd

def get_tables_names(db_path):
    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Execute a query to retrieve all table names
    cursor.execute('''SELECT name FROM sqlite_master WHERE type='table';''')
    tables = cursor.fetchall()

    conn.close()
    return tables

def print_tables_contents(tables, db_path):
    conn = sqlite3.connect(db_path)

    # Load all tables into pandas DataFrames and print their contents
    for table_name in tables:
        table_name = table_name[0]
        df = pd.read_sql_query(f'SELECT * FROM {table_name} LIMIT 1;', conn)
        print(f"Table: {table_name}")
        print(df)
        print("\n")
    
    conn.close()

if __name__ == "__main__":
    db_path = 'database.sqlite'
    tables = get_tables_names(db_path)
    print_tables_contents(tables, db_path)

    db_path_clean = 'cleandataset.sqlite'
    tables_clean = get_tables_names(db_path_clean)
    print_tables_contents(tables_clean, db_path_clean)