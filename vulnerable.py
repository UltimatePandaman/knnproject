#SRC: https://snyk.io/blog/code-injection-python-prevention-examples/

user_input = input("Enter filename: ")
with open(user_input, 'r') as file:  # Vulnerable to directory traversal
    content = file.read()
    
user_input = input("Enter your username: ")
query = "SELECT * FROM users WHERE username = '" + user_input + "';"
execute_query(query)  # This can be exploited

import os

directory = input("Enter the directory to list: ")
command = f"ls {directory}"  # Vulnerable to Command Injection
os.system(command)

import pickle
serialized_data = input("Enter serialized data: ")
deserialized_data = pickle.loads(serialized_data.encode('latin1'))  # Unsafe deserialization

ALLOWED_COMMANDS = ["start", "stop", "restart"]
user_input = input("Enter your command: ")
if user_input in ALLOWED_COMMANDS:
    exec(user_input)
else:
    print("Invalid command.")
    
from ast import literal_eval
user_input = input("Enter expression: ")
try:
    result = literal_eval(user_input)
except ValueError:
    print("Invalid expression.")

import re
user_input = input("Enter filename: ")
if re.match("^[a-zA-Z0-9_\-/]+\.txt$", user_input):
    with open(user_input, 'r') as file:
        content = file.read()
else:
    print("Invalid filename.")
    
    
import sqlite3
connection = sqlite3.connect('database.db')
cursor = connection.cursor()
input_username = input("Enter username: ")
query = "SELECT * FROM users WHERE username = ?"
cursor.execute(query, (input_username,))

import json
serialized_data = input("Enter serialized data: ")
try:
    deserialized_data = json.loads(serialized_data)
except json.JSONDecodeError:
    print("Invalid data.")
