"""Render the first page of paper.pdf to PNG, and crop the first figure."""
import fitz  # PyMuPDF
import os

PDF = r"E:\CloudStation\MosaicOpt\docs\SIGMOD\paper.pdf"
OUT_DIR = r"E:\CloudStation\MosaicOpt\docs\SIGMOD\figures"
os.makedirs(OUT_DIR, exist_ok=True)

doc = fitz.open(PDF)
page = doc[0]
print(f"Page 1 size: {page.rect}")

# Render full page 1 at 200 DPI
mat = fitz.Matrix(200 / 72, 200 / 72)
pix = page.get_pixmap(matrix=mat, alpha=False)
full_path = os.path.join(OUT_DIR, "page1_full.png")
pix.save(full_path)
print(f"Saved full page 1: {full_path} ({pix.width}x{pix.height})")

# The first figure occupies roughly the top of page 1.
# In ACM SIGCONF two-column, the figure spans the full page width.
# Render at 200 DPI then crop the top 50% of the page.
full_w, full_h = pix.width, pix.height

# Crop just the figure area (right column, from below the title block
# through the end of the caption).
page_rect = page.rect
fig_clip = fitz.Rect(308, 180, page_rect.width, 358)
mat_crop = fitz.Matrix(260 / 72, 260 / 72)
pix_cropped = page.get_pixmap(matrix=mat_crop, clip=fig_clip, alpha=False)
crop_path = os.path.join(OUT_DIR, "figure1.png")
pix_cropped.save(crop_path)
print(f"Saved cropped figure 1: {crop_path} ({pix_cropped.width}x{pix_cropped.height})")

doc.close()
