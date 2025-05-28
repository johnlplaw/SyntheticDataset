import mysql.connector

# Create connection
def get_mysql_conn():
    mydb = mysql.connector.connect(
        host="localhost",
        user="root",
        password="myroot",
        database="syntheticDS"
    )
    return mydb
