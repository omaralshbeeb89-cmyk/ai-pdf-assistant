"""
Terminal version of the AI PDF Assistant.

    python app.py                     # default PDF
    python app.py pdf/other.pdf       # another PDF
    python app.py a.pdf b.pdf         # several PDFs together
    python app.py --debug             # show retrieval details

For the web version run:  streamlit run streamlit_app.py
"""

import sys

from pdf_assistant.cli import main


if __name__ == "__main__":
    sys.exit(main())
