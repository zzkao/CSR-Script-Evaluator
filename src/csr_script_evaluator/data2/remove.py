#!/usr/bin/env python3
"""
remove_between.py

For each .sh file in current directory:
 - If it contains a shebang line (starts with #!) and a line that begins with "# Training"
 - Remove everything between those two lines, keeping both boundary lines.
 - Create a backup file <name>.bak before overwriting.
"""

import re
from pathlib import Path

MARKER_RE = re.compile(r'^\s*#\s*Training\b', flags=re.I)  # case-insensitive marker match

def process_text(text):
    # Find first shebang line (line that starts with #!)
    lines = text.splitlines(keepends=True)
    shebang_idx = None
    training_idx = None

    for i, ln in enumerate(lines):
        if ln.startswith('#!'):
            shebang_idx = i
            break

    if shebang_idx is None:
        return None  # no change: no shebang

    for j in range(shebang_idx + 1, len(lines)):
        if MARKER_RE.match(lines[j]):
            training_idx = j
            break

    if training_idx is None:
        return None  # no change: no training marker after shebang

    # Keep shebang line and the training line + everything after training.
    new_lines = []
    new_lines.append(lines[shebang_idx])
    new_lines.append('\n' if lines[shebang_idx].endswith('\n') and (len(lines) > shebang_idx+1 and lines[shebang_idx+1].strip() == '') else '')
    # Ensure there is exactly one blank line between shebang and training (optional)
    # Append the training line and the rest
    new_lines.append(lines[training_idx])
    new_lines.extend(lines[training_idx + 1 :])
    return ''.join(new_lines)

def main():
    cwd = Path('.')
    sh_files = list(cwd.glob('*.sh'))
    if not sh_files:
        print("No .sh files found in current directory.")
        return

    for p in sh_files:
        txt = p.read_text(encoding='utf-8')
        new_txt = process_text(txt)
        if new_txt is None:
            print(f"Skipping {p.name} (missing shebang or '# Training' after it).")
            continue

        bak = p.with_suffix(p.suffix + '.bak')
        bak.write_text(txt, encoding='utf-8')
        p.write_text(new_txt, encoding='utf-8')
        print(f"Updated {p.name}  (backup {bak.name})")

if __name__ == '__main__':
    main()
