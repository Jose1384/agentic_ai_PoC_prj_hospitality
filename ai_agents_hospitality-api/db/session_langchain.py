
from langchain_community.utilities import SQLDatabase
from config.agent_config import _get_env_value

def get_database() -> SQLDatabase:
    """
    Create and return the PostgreSQL database connection.
    """
    return SQLDatabase.from_uri(
        _get_env_value("DB_URI")
    )

# Check db conn and print sample data
#db_global = get_database()
#print("Database connection established.")
#
## Probar ejecución de consulta simple
#try:
#    result = db_global.run("SELECT 1;")
#    tables = db_global.run("""
#SELECT table_name
#FROM information_schema.tables
#WHERE table_schema = 'public';
#""")
#    rows = db_global.run("SELECT * FROM bookings LIMIT 10;")
#    print("Sample rows from bookings:")
#    for row in rows:
#        print(row)
#    print("✅ Connection test successful. Query result:", result)
#except Exception as e:
#    print("❌ Connection test failed:", e)