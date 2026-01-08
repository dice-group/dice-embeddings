import argparse

def compare_files(path_a: str, path_b: str) -> None:
    with open(path_a, "r", encoding="utf-8", errors="replace") as fa:
        a_lines = [line.rstrip("\n") for line in fa]

    with open(path_b, "r", encoding="utf-8", errors="replace") as fb:
        b_lines = [line.rstrip("\n") for line in fb]

    same = 0
    diff = 0

    n = min(len(a_lines), len(b_lines))
    for i in range(n):
        if a_lines[i] == b_lines[i]:
            same += 1
        else:
            diff += 1


    print(f"File A lines: {len(a_lines)}")
    print(f"File B lines: {len(b_lines)}")
    print(f"Same rows:     {same}")
    print(f"Different rows:{diff}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare two text files line-by-line and count same/different rows."
    )
    parser.add_argument("file_a", help="First text file")
    parser.add_argument("file_b", help="Second text file")
    args = parser.parse_args()

    compare_files(args.file_a, args.file_b)

if __name__ == "__main__":
    main()
