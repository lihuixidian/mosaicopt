#!/usr/bin/env python3
"""Humanize pass 2: handle remaining em-dash contexts."""
import re

PATH = r'E:\CloudStation\MosaicOpt\docs\SIGMOD\paper.tex'

with open(PATH, 'r', encoding='utf-8') as f:
    content = f.read()

lines = content.split('\n')
out = []

# Per-line replacement rules
def fix_em_dashes(line):
    """Apply contextual replacements."""
    original = line

    # Pattern 1: leading em dash (after whitespace)
    # "   --- text" -> "   text" (treating as continuation of previous sentence/para)
    # If the line starts (after whitespace) with `---` and previous line ended without period,
    # it should become just a comma-separated continuation.
    m = re.match(r'^(\s*)---\s+(.*)$', line)
    if m:
        indent, rest = m.group(1), m.group(2)
        # If the rest is a complete sentence (starts with capital + verb),
        # treat as new sentence after "; "
        # Otherwise it's a list item / continuation
        if rest and rest[0].isupper():
            line = indent + '; ' + rest
        else:
            line = indent + rest

    # Pattern 2: em dash after math `$` -> `, `
    # "schema --- that" already fixed in pass 1
    # Pattern: \Mem\ --- word
    line = re.sub(r'(\\\w+\\) --- ', r'\1, ', line)
    # Pattern: $...$ --- word
    line = re.sub(r'(\$[^$]*\$) --- ', r'\1, ', line)
    # Pattern: \text{...} --- text  (already handled by previous but might miss)
    line = re.sub(r'(\\text\{[^}]+\}) --- ', r'\1, ', line)

    # Pattern 3: em dash at end of sentence fragment continuing onto next line
    # Already handled.

    # Pattern 4: "production ---" at end of line, continues next line as parenthetical
    # "The four use cases that dominate Agentic memory in production ---"
    # Then "assembly, and tool-trace analysis --- are all conjunctive"
    # We need to fix both lines together, but for simplicity, let's just remove the trailing dash
    # and rely on the next line's leading context.
    # Actually, just replace trailing `---` with `:` so it acts as colon
    if line.rstrip().endswith('---'):
        line = line.rstrip()[:-3].rstrip() + ':'

    # Pattern 5: "--- its search is over..." -> ", its search is over..."
    line = re.sub(r'\s+---\s+(?=(its|the|a|an|this|that|these|those|here|where)\b)', r'; \1', line)

    # Pattern 6: standalone `--- Theorem~\ref` -> comma `, Theorem~\ref`
    line = re.sub(r'\s+---\s+(Theorem~\\ref)', r'; \1', line)

    # Pattern 7: chained cases (something), (something) --- (something)
    # If line ends with em dash and previous comma'd items, leave colon
    # Already handled by `endswith('---')` above.

    return line

# Process
for line in lines:
    stripped = line.lstrip()
    if stripped.startswith('%'):
        out.append(line)
        continue
    # Skip lines inside tikzpicture
    if '\\begin{tikzpicture}' in line:
        out.append(line)
        continue
    new_line = fix_em_dashes(line)
    out.append(new_line)

new_content = '\n'.join(out)

# Post-process: collapse stray spaces around punctuation
new_content = re.sub(r' +([,.;:?!])', r'\1', new_content)
# Replace `,,` or `;;` (double punctuation) - shouldn't happen but just in case
new_content = re.sub(r',,+', r',', new_content)
new_content = re.sub(r';;+', r';', new_content)
new_content = re.sub(r'\.\.+', r'.', new_content)

# Clean up ", ;" -> ";" and "; ," -> "," from naive replacement
new_content = re.sub(r', ; ', '; ', new_content)
new_content = re.sub(r'; , ', ', ', new_content)

with open(PATH, 'w', encoding='utf-8') as f:
    f.write(new_content)

remaining = new_content.count('---')
print(f'Pass 2 done. Remaining em dashes: {remaining}')