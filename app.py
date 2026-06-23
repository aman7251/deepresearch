"""Top-level entrypoint for Hugging Face Spaces.

Spaces launches a Streamlit app by running `streamlit run app.py`. Locally you can
use this too, or run `streamlit run app/ui_streamlit.py` directly.
"""
from app.ui_streamlit import main

main()
