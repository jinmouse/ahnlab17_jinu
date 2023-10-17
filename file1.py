def read_review_file(filename):
    """
    Reads the content of a file and returns it.
    """
    with open(filename, 'r', encoding="utf-8") as file:
        content = file.read()
    return content

def main():
    for i in range(1, 6):
        filename = f"data/review{i}.txt"
        print(f"Contents of {filename}:\n")
        print(read_review_file(filename))
        print("=" * 50)  # prints a separator

if __name__ == "__main__":
    main()
