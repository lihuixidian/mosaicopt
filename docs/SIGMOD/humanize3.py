#!/usr/bin/env python3
"""Humanize pass 3: handle em dashes inside \text{} in math."""
import re

PATH = r'E:\CloudStation\MosaicOpt\docs\SIGMOD\paper.tex'

with open(PATH, 'r', encoding='utf-8') as f:
    content = f.read()

# Pattern: \text{--- text} -> \text{(text)}  or \text{: text}
# Use parens to clearly delimit
def repl(m):
    inner = m.group(1)
    # inner starts with `--- ` and then has the actual text
    if inner.startswith('--- '):
        rest = inner[4:]
        return f'\\text{{({rest})}}'
    if inner.startswith('---'):
        rest = inner[3:]
        return f'\\text{{({rest})}}'
    return m.group(0)

new_content = re.sub(r'\\text\{--- ([^}]+)\}', repl, content)
# Also handle cases where `---` is followed by space, then non-space
new_content = re.sub(r'\\text\{---([^}]+)\}', repl, new_content)

with open(PATH, 'w', encoding='utf-8') as f:
    f.write(new_content)

remaining = new_content.count('---')
print(f'Pass 3 done. Remaining em dashes: {remaining}')