from pathlib import Path

EXCLUDE = {".venv", ".github", "__pycache__", ".git"}

def write_tree(directory, file, prefix=""):
    entries = sorted(
        [e for e in directory.iterdir() if e.name not in EXCLUDE],
        key=lambda x: (x.is_file(), x.name.lower())
    )

    for i, entry in enumerate(entries):
        is_last = i == len(entries) - 1

        connector = "\\-- " if is_last else "|-- "
        file.write(prefix + connector + entry.name + "\n")

        if entry.is_dir():
            extension = "    " if is_last else "|   "
            write_tree(entry, file, prefix + extension)

root = Path(".")

with open("structure.txt", "w", encoding="utf-8") as f:
    f.write(root.resolve().name + "\n")
    write_tree(root, f)

print("Done! Structure written to structure.txt")