user_input = input("Enter expression: ")
result = eval(user_input)  # Unsafe

password = input("Enter expression: ")
if password == "supersecretpassword101":
    print("Access granted")