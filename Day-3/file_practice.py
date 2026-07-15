#Practice Problem 1: Create a text file and write data into it.
with open("sample.txt", "w") as f:
    f.write("This is line one.\n")
    f.write("This is line two.\n")

#Practice Problem 2: Read and display file contents.
with open("sample.txt", "r") as f:
    content = f.read()
    print(content)

#Practice Problem 3: Append new data to an existing file.
with open("sample.txt", "a") as f:
    f.write("This is line three.\n")

#Practice Problem 4: Count the number of lines in a file.
with open("sample.txt", "r") as f:
    lines = f.readlines()
    print(f"Number of lines: {len(lines)}")

