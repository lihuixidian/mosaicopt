"""High-resolution render of the application layer area only."""
import fitz

PDF = r"E:\CloudStation\MosaicOpt\docs\SIGMOD\paper.pdf"
OUT = r"E:\CloudStation\MosaicOpt\docs\SIGMOD\figures\figure1_hd.png"

doc = fitz.open(PDF)
page = doc[0]

# The figure is at the top of page 1, in the right column.
# Page is 612x792 pt. Right column starts around x=315.
# Figure starts around y=200 (after title block).
# Application layer is taller now (3.0cm), so extends to ~y=380.

# Let's render the full right column at high DPI
clip = fitz.Rect(310, 200, 605, 400)
mat = fitz.Matrix(400 / 72, 400 / 72)
pix = page.get_pixmap(matrix=mat, clip=clip, alpha=False)
pix.save(OUT)
print(f"Saved HD: {OUT} ({pix.width}x{pix.height})")
doc.close()
