import bcrypt

passwords_to_hash = ['abc', '123']

for pwd in passwords_to_hash:
    # Generate a salt and hash the password
    hashed_pwd = bcrypt.hashpw(pwd.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    print(f"Hash for '{pwd}': {hashed_pwd}")
