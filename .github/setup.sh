python -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
git clone https://github.com/descriptinc/audiotools
python -m pip install audiotools
python -m pip install -e .
rm -rf audiotools
python -m pip install --upgrade streamlit
