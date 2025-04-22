#!/usr/bin/env python3
# Author: Theeko74
# Contributor(s): skjerns
# Oct, 2021
# MIT license -- free to use as you want, cheers.

"""
Simple python wrapper script to use ghoscript function to compress PDF files.

Compression levels:
    0: default - almost identical to /screen, 72 dpi images
    1: prepress - high quality, color preserving, 300 dpi imgs
    2: printer - high quality, 300 dpi images
    3: ebook - low quality, 150 dpi images
    4: screen - screen-view-only quality, 72 dpi images

Dependency: Ghostscript.
On MacOSX install via command line `brew install ghostscript`.
"""
import os.path
import shutil
import sys
import ghostscript
import locale

def compress_pdf(input_file_path, output_file_path, power=0):
    """Compress PDF using Ghostscript Python library."""
    quality = {
        0: "/default",
        1: "/prepress",
        2: "/printer",
        3: "/ebook",
        4: "/screen"
    }

    if not os.path.isfile(input_file_path):
        raise FileNotFoundError(f"Input file '{input_file_path}' does not exist.")

    if power not in quality:
        raise ValueError(f"Invalid compression level: {power}. Valid levels are 0 to 4.")

    args = [
        "gs",
        "-sDEVICE=pdfwrite",
        "-dCompatibilityLevel=1.4",
        f"-dPDFSETTINGS={quality[power]}",
        "-dNOPAUSE",
        "-dQUIET",
        "-dBATCH",
        f"-sOutputFile={output_file_path}",
        input_file_path
    ]

    # Encode arguments for Ghostscript
    encoding = locale.getpreferredencoding()
    args = [a.encode(encoding) for a in args]

    try:
        ghostscript.Ghostscript(*args)
        print(f"Compressed '{input_file_path}' to '{output_file_path}'.")
    except ghostscript.GhostscriptError as e:
        print(f"An error occurred during compression: {e}")
        sys.exit(1)

def compress(input_path, output_path=None, compress_level=2, backup=False):
    """
    Compresses a PDF file using Ghostscript.

    :param input_path: Path to the input PDF file.
    :param output_path: Path to the output PDF file. Defaults to overwriting the input file.
    :param compress_level: Compression level from 0 to 4. Defaults to 2.
    :param backup: Whether to create a backup of the original PDF. Defaults to False.
    """
    if not output_path:
        output_path = "temp.pdf"

    compress_pdf(input_path, output_path, power=compress_level)

    if output_path == "temp.pdf":
        if backup:
            backup_path = input_path.replace(".pdf", "_BACKUP.pdf")
            shutil.copyfile(input_path, backup_path)
            print(f"Backup created at '{backup_path}'.")
        shutil.copyfile(output_path, input_path)
        os.remove(output_path)
        print(f"Overwritten original file with compressed version.")

