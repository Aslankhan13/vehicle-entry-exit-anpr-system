import sqlite3

# Connect to database
conn = sqlite3.connect('database.db')
cursor = conn.cursor()

# Create the LicensePlates table if it doesn't exist
cursor.execute('''
    CREATE TABLE IF NOT EXISTS LicensePlates(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        start_time TEXT,
        end_time TEXT,
        license_plate TEXT
    )
''')

# Commit and close
conn.commit()
conn.close()
