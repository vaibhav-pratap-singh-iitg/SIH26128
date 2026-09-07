"""
=============================================================================
  Livestock Health Early Warning System — Google Colab Hosting Guide
=============================================================================

  Copy-paste each numbered cell block below into separate Colab cells.
  Run them in order (Cell 1 → Cell 7). Your Streamlit app will be live
  on a public ngrok URL within ~2 minutes.

  PREREQUISITES:
    1. Upload 'livestock-ai-mvp.zip' to your Colab session (Files panel).
    2. Get a FREE ngrok authtoken from https://dashboard.ngrok.com/signup
       (one-click Google sign-in, then copy the token from the dashboard).

  *** ALL DATA IS SYNTHETIC — see README.md. ***
=============================================================================
"""

# ═══════════════════════════════════════════════════════════════════════════
# CELL 1  —  Unzip and navigate to project root
# ═══════════════════════════════════════════════════════════════════════════
# The zip extracts to livestock-ai-mvp/livestock-ai-mvp/ (nested).
# We need to be inside the INNER directory (where app.py lives).
#
# Paste into a Colab cell:
# ─────────────────────────────────────────────────────────────────────────
#   !unzip -qo /content/livestock-ai-mvp.zip -d /content/
#   %cd /content/livestock-ai-mvp/livestock-ai-mvp
#   !ls app.py src/ data/ models/
# ─────────────────────────────────────────────────────────────────────────
# You should see:  app.py  src/  data/  models/
# If 'app.py' is not listed, the zip has a different structure —
# adjust the %cd path accordingly.


# ═══════════════════════════════════════════════════════════════════════════
# CELL 2  —  Install Python dependencies
# ═══════════════════════════════════════════════════════════════════════════
# Paste into a Colab cell:
# ─────────────────────────────────────────────────────────────────────────
#   !pip install -q -r requirements.txt pyngrok
# ─────────────────────────────────────────────────────────────────────────


# ═══════════════════════════════════════════════════════════════════════════
# CELL 3  —  Generate synthetic data
# ═══════════════════════════════════════════════════════════════════════════
# Paste into a Colab cell:
# ─────────────────────────────────────────────────────────────────────────
#   !python src/data_generator.py
# ─────────────────────────────────────────────────────────────────────────
# Expected output: "Generated 90 regions" and "Generated ~5400 reports".


# ═══════════════════════════════════════════════════════════════════════════
# CELL 4  —  Train the ML model
# ═══════════════════════════════════════════════════════════════════════════
# Paste into a Colab cell:
# ─────────────────────────────────────────────────────────────────────────
#   !python src/train_model.py
# ─────────────────────────────────────────────────────────────────────────
# Expected output: accuracy metrics + "Saved model -> models/risk_model.pkl"


# ═══════════════════════════════════════════════════════════════════════════
# CELL 5  —  Run tests (optional but recommended)
# ═══════════════════════════════════════════════════════════════════════════
# Paste into a Colab cell:
# ─────────────────────────────────────────────────────────────────────────
#   !pytest -q
# ─────────────────────────────────────────────────────────────────────────
# Expected: all tests pass (7 passed).


# ═══════════════════════════════════════════════════════════════════════════
# CELL 6  —  Configure ngrok authtoken
# ═══════════════════════════════════════════════════════════════════════════
# Replace YOUR_NGROK_TOKEN_HERE with your token from
# https://dashboard.ngrok.com/get-started/your-authtoken
#
# Paste into a Colab cell:
# ─────────────────────────────────────────────────────────────────────────
#   !ngrok config add-authtoken YOUR_NGROK_TOKEN_HERE
# ─────────────────────────────────────────────────────────────────────────


# ═══════════════════════════════════════════════════════════════════════════
# CELL 7  —  Launch Streamlit + ngrok tunnel
# ═══════════════════════════════════════════════════════════════════════════
# This cell will print a public URL (e.g. https://xxxx.ngrok-free.app).
# Open that URL in your browser to see the dashboard.
#
# Paste into a Colab cell:
# ─────────────────────────────────────────────────────────────────────────
#   import subprocess
#   import time
#   from pyngrok import ngrok
#
#   # Kill any leftover Streamlit processes
#   !pkill -f streamlit || true
#
#   # Launch Streamlit in the background
#   proc = subprocess.Popen(
#       ["streamlit", "run", "app.py",
#        "--server.headless", "true",
#        "--server.enableCORS", "false",
#        "--server.enableXsrfProtection", "false",
#        "--server.port", "8501"],
#       stdout=open("/content/streamlit.log", "w"),
#       stderr=subprocess.STDOUT,
#   )
#
#   # Wait for Streamlit to start
#   time.sleep(5)
#
#   # Open ngrok tunnel
#   public_url = ngrok.connect(8501)
#   print("=" * 60)
#   print(f"🐄  LIVESTOCK DASHBOARD IS LIVE AT:")
#   print(f"    {public_url}")
#   print("=" * 60)
#   print("(Keep this cell running. To stop, interrupt the cell.)")
# ─────────────────────────────────────────────────────────────────────────


# ═══════════════════════════════════════════════════════════════════════════
# ALTERNATIVE CELL 7  —  If you don't have an ngrok token, use localtunnel
# ═══════════════════════════════════════════════════════════════════════════
# NOTE: localtunnel sometimes shows a "click to continue" page.
#       Enter the Colab server IP shown in the page to bypass it.
#
# Paste into a Colab cell:
# ─────────────────────────────────────────────────────────────────────────
#   !pkill -f streamlit || true
#   !streamlit run app.py \
#       --server.headless true \
#       --server.enableCORS false \
#       --server.enableXsrfProtection false \
#       --server.port 8501 \
#       &>/content/streamlit.log &
#
#   import time; time.sleep(5)
#
#   # Get the server IP (needed for localtunnel's verification page)
#   import urllib.request
#   ip = urllib.request.urlopen("https://ipv4.icanhazip.com").read().decode().strip()
#   print(f"⚠️  If localtunnel shows a verification page, enter this IP: {ip}")
#
#   !npx -y localtunnel --port 8501
# ─────────────────────────────────────────────────────────────────────────


# ═══════════════════════════════════════════════════════════════════════════
# TROUBLESHOOTING
# ═══════════════════════════════════════════════════════════════════════════
#
# Problem: Blank/white screen after opening the tunnel URL
# Fix:     The .streamlit/config.toml file we added disables CORS/XSRF.
#          If you still see a blank screen, make sure Cells 1-6 ran without
#          errors and try the command-line flags in Cell 7.
#
# Problem: "ModelNotTrainedError" even after running train_model.py
# Fix:     Make sure Cell 3 (data generator) and Cell 4 (training) both
#          completed successfully. Check that models/risk_model.pkl exists:
#            !ls -la models/
#
# Problem: "No module named 'src'"
# Fix:     Your working directory is wrong. Run:
#            %cd /content/livestock-ai-mvp/livestock-ai-mvp
#          and verify with `!ls app.py`
#
# Problem: localtunnel "click to continue" interstitial
# Fix:     Enter the server IP printed above the tunnel URL. Or switch to
#          ngrok (Cell 6 + Cell 7 main version) which doesn't have this issue.
#
# Problem: Tests fail with import errors
# Fix:     We've added tests/__init__.py. Make sure you're using the fixed
#          version of the project.
