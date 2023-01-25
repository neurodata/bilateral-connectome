from svglib.svglib import svg2rlg
from reportlab.graphics import renderPDF


def svg_to_pdf(svg_file, pdf_file):
    drawing = svg2rlg(svg_file)
    renderPDF.drawToFile(drawing, str(pdf_file))
