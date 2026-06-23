# Deploying DeepResearch (free tiers)

Goal: a public demo link that works **without anyone entering API keys**. The
simplest path uses **demo mode** (a cached run); the full app needs Redis + a worker.

## Option 1 — Hugging Face Spaces (Streamlit, demo mode)

Best for a zero-cost, always-on public link.

1. Create a new **Streamlit** Space.
2. Push this repo to the Space (or connect the GitHub repo).
3. In the Space **Settings → Variables**, set `DEMO_MODE=1`.
4. The Space runs `streamlit run app/ui_streamlit.py` and serves the cached sample run.

> For live (non-demo) research on a Space you also need Redis + a worker, which a
> single free Space can't host. Use demo mode for the public link and run the full
> stack locally / on a VPS for live demos.

## Option 2 — cloudflared tunnel (live, from your laptop)

Expose the locally-running Streamlit app over a public HTTPS URL, free:

```bash
docker compose up --build            # app on :8501
cloudflared tunnel --url http://localhost:8501
```

Cloudflare prints a temporary `https://*.trycloudflare.com` URL.

## Option 3 — persistent box (always-free / cheap VPS)

- **Oracle Cloud Always-Free** ARM VM or a small Hetzner box.
- Install Docker, clone the repo, set `.env` (with `GROQ_API_KEY`), `docker compose up -d`.
- Put the Streamlit port behind a reverse proxy (Caddy/Nginx) for TLS.

## Notes
- The demo Space needs **no secrets**. For live mode keep `GROQ_API_KEY` in the host's
  secret store / env — never commit it (`.env` is gitignored).
- `runs/` and the HuggingFace model cache are mounted as Docker volumes so they persist
  across restarts.

---

## Pushing to GitHub (personal, isolated from any work git identity)

Set a **per-repo** identity so your global (possibly work) git config is untouched:

```bash
cd C:\personal_project\deepresearch
git init
git config user.name  "aman7251"
git config user.email "ranaamansingh7251@gmail.com"
git add .
git commit -m "feat: DeepResearch MVP"
git branch -M main
git remote add origin https://github.com/aman7251/deepresearch.git
git push -u origin main      # username: aman7251 ; password: your fine-grained PAT
```

> Rotate any token you have shared in plaintext. Use a fine-grained PAT scoped to this
> repo with **Contents: Read and write**.
