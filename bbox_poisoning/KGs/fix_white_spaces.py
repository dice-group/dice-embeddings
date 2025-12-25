import sys

def fix_whitespace(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            # strip newline but keep empty detection
            raw = line.rstrip("\n")
            if not raw.strip():
                # keep blank lines if you want, or skip with "continue"
                continue

            # split on ANY whitespace (spaces, tabs, multiple spaces, etc.)
            parts = raw.split()

            # If a line has more than 3 tokens, keep first 3.
            # If it has fewer (like your falkland_islands line), we just join what exists.
            cleaned = "    ".join(parts[:3])

            fout.write(cleaned + "\n")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} INPUT_FILE OUTPUT_FILE")
        sys.exit(1)

    fix_whitespace(sys.argv[1], sys.argv[2])
