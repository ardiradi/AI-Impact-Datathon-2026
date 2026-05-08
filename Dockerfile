# ── Base Image ────────────────────────────────────────────────────────────────
FROM python:3.9

# ── Hugging Face Spaces: buat user non-root ──────────────────────────────────
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

# ── Working Directory ────────────────────────────────────────────────────────
WORKDIR /app

# ── Install Dependencies ─────────────────────────────────────────────────────
COPY --chown=user ./requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# ── Copy Application Files ───────────────────────────────────────────────────
COPY --chown=user . /app

# ── Expose Port (Hugging Face Spaces menggunakan 7860) ───────────────────────
EXPOSE 7860

# ── Run Server ───────────────────────────────────────────────────────────────
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]