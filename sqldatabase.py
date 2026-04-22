import sqlite3

conn = sqlite3.connect('database.db')

cursor = conn.cursor()


cursor.execute(
    '''
    CREATE TABLE IF NOT EXISTS LicensePlates(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        start_time TEXT,
        end_time TEXT,
        license_plate TEXT
     )
    '''
)