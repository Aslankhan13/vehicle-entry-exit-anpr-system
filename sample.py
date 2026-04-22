import mysql.connector

db = mysql.connector.connect(
    host = "localhost",
    user = "aslan",
    password = "12345",
    database = "vehicle_management"
)

cursor = db.cursor()
print ("Connected to MySql DB")

cursor.execute("SELECT * FROM entrylog")
for row in cursor.fetchall():
    print(row)