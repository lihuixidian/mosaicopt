#!/usr/bin/env python3
"""
Humanize paper.tex: replace AI-typical em dashes (---) with appropriate
punctuation while preserving math/commands/labels/citations.

Strategy:
- Skip lines that start with % (LaTeX comments)
- Skip lines inside tikzpicture environments
- Replace ` --- ` with contextual punctuation:
  - `---` between two phrases (appositive) -> `: ` or `, `
  - `---` before a sentence -> `; ` or `. `
  - `---` in theorem names (`\begin{theorem}[X --- Y]`) -> `: `
- Do not touch `\text{--- ...}` (text-mode em dash inside math)
"""
import re
import sys

PATH = r'E:\CloudStation\MosaicOpt\docs\SIGMOD\paper.tex'

# Read file
with open(PATH, 'r', encoding='utf-8') as f:
    content = f.read()

# We'll process line by line, tracking tikzpicture environments
lines = content.split('\n')
out = []
in_tikz = False
inside_text_dash = 0  # tracker for \text{--- ...}

# Em-dash punctuation lookup table for prose replacements
# Format: (pattern_before, pattern_after) -> replacement
# We do this per-line to keep it simple

def looks_like_appositive(line, idx):
    """Check if ` --- ` at idx is followed by a noun-ish clause (appositive)."""
    after = line[idx+4:idx+60].strip()
    # If after starts with lowercase or article, likely appositive -> `, `
    # If after starts with capitalized word that's not a new sentence, also appositive
    # Heuristic: if the line BEFORE ` --- ` doesn't end with `.`, it's appositive
    before = line[:idx].rstrip()
    if before.endswith('.'):
        return False  # new sentence -> semicolon or period
    return True

def replace_dash_in_line(line):
    """Replace ` --- ` in a prose line."""
    new = line
    while True:
        idx = new.find(' --- ')
        if idx < 0:
            break
        # Check context
        after = new[idx+5:idx+5+60]
        before = new[:idx].rstrip()

        # Decide replacement
        # If after starts with a capital letter and a verb-like construction (likely new sentence),
        # use '; ' or '. '
        # If after is an appositive / clarification / elaboration, use ', '

        # Special: if before ends with comma, semicolon, or parens -> just space-comma
        if before.endswith(','):
            rep = ', '
        elif before.endswith(';'):
            rep = ', '
        elif before.endswith('('):
            # parenthesis opening -> just space after paren or remove leading dash
            rep = ' '  # "open paren --- text" -> "open paren text"
        elif before.endswith(':'):
            rep = ', '
        elif before.endswith('.'):
            # new sentence fragment -> semicolon
            rep = '; '
        elif before.endswith('\$'):
            rep = ', '
        else:
            # Appositive case -> comma
            rep = ', '
        new = new[:idx] + rep + new[idx+5:]
    return new

def replace_dash_theorem_name(line):
    """Replace `---` inside `\begin{theorem}[X --- Y]` with `:`."""
    # Pattern: \begin{theorem}[... --- ...]
    pattern = re.compile(r'(\\begin\{theorem\}\[[^\]]*?) --- ')
    def repl(m):
        return m.group(1) + ': '
    return pattern.sub(repl, line)

# Process
processed_lines = []
for i, line in enumerate(lines):
    stripped = line.lstrip()
    # Track TikZ
    if '\\begin{tikzpicture}' in line:
        in_tikz = True
        processed_lines.append(line)
        continue
    if '\\end{tikzpicture}' in line:
        in_tikz = False
        processed_lines.append(line)
        continue
    if in_tikz:
        processed_lines.append(line)
        continue
    # Skip comments (only if line starts with %)
    if stripped.startswith('%'):
        processed_lines.append(line)
        continue
    # Skip lines that are entirely inside math display? No, we still process them.

    # Replace theorem names
    new_line = replace_dash_theorem_name(line)
    # Replace prose em dashes
    new_line = replace_dash_in_line(new_line)

    processed_lines.append(new_line)

new_content = '\n'.join(processed_lines)

# Special: clean up some obvious artifacts
# Replace ", ;" and "; ," introduced by naive replacement
new_content = re.sub(r', ; ', '; ', new_content)
new_content = re.sub(r'; , ', ', ', new_content)
# Replace "  " (double space) with " " after punctuation if needed
new_content = re.sub(r' +([,.;:?!])', r'\1', new_content)
# But preserve intentional indentation; only fix at end-of-line / mid-prose
new_content = re.sub(r'([a-z])  +([a-z])', r'\1 \2', new_content)

with open(PATH, 'w', encoding='utf-8') as f:
    f.write(new_content)

# Count remaining em dashes
remaining = new_content.count('---')
print(f'Processed. Remaining em dashes: {remaining}')