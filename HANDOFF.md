# DeepResearch — Handoff & Completion Guide

> Paste this whole file into a fresh AI chat on the **personal laptop**. It is
> self-contained: an agent with none of the prior context can finish the loose ends,
> test everything, and deploy a public demo from here.

---

## 0. TL;DR for the personal-laptop agent

The project is **fully written and committed locally** (2 commits on `main`) but was
scaffolded on a restricted network where `pip`, Docker, model downloads, and the GitHub
push were all blocked. Your job:

1. Get the code onto this laptop (transfer the folder, or clone if it was already pushed).
2. Create a Python venv and install deps.
3. Run the offline test suite (`pytest`) — this proves the whole agent loop works with **no keys**.
4. Push to GitHub `aman7251/deepresearch` (rotate the old token first).
5. Add a Groq key (or Ollama) and do a live run.
6. Deploy a public demo (Hugging Face Spaces in demo mode is the fastest).

Acceptance criteria are in section 9.

---

## 1. Context & hard constraints

- **Owner GitHub:** `aman7251` (personal, public repos). Repo: <https://github.com/aman7251/deepresearch>
- **Commit identity (per-repo, NEVER touch global git config):**
  `user.name = aman7251`, `user.email = ranaamansingh7251@gmail.com`
- **Open-source / free only.** No company resources, keys, code, or cloud. Everything runs
  locally or on free / personal tiers.
- **Machine:** personal laptop on home network (no corporate proxy).
- **OS:** Windows (PowerShell). Adapt commands for macOS/Linux if needed.

## 2. What the project is

An autonomous multi-agent research assistant with a **fact-checking verifier**. You ask a
hard question; a LangGraph loop runs **planner → web-search → reader → synthesizer →
critic/verifier**, builds an ephemeral RAG index over fetched pages, writes a **cited
report**, and the verifier fact-checks every claim against its sources
(`supported` / `unsupported` / `contradicted` + confidence), looping back once to revise.
Ships with token/cost observability, demo mode, and an eval + prompt-injection red-team.

### Stack
- Orchestration: LangGraph (model-agnostic; deepagents-compatible)
- LLM: **Groq** free tier (default) or **Ollama** (local) — both OpenAI-compatible
- Web search: `ddgs` (no key) · Reader: `httpx` + `trafilatura`
- RAG: embedded **Chroma** + `sentence-transformers` (BGE-small)
- API/queue: FastAPI + Redis + `arq` worker · UI: Streamlit
- Tests: pytest (fully offline, faked LLM/search/index) · CI: GitHub Actions

### Current state
- Branch `main`, commits: `feat: DeepResearch MVP ...` and `test: offline pytest suite, CI ...`
- Remote `origin` already set to `https://github.com/aman7251/deepresearch.git`
- **Not yet pushed.** Deps **not** installed. No live run done yet.

## 3. Get the code onto this laptop

**Preferred — transfer the existing folder** (it keeps the git history + local commits):
copy the whole `deepresearch` folder (e.g. via USB, OneDrive, or a zip) to this laptop,
then `cd` into it. Confirm history:
```powershell
git log --oneline    # should show the 2 commits above
```

**Alternative — if the repo was already pushed:** just clone it:
```powershell
git clone https://github.com/aman7251/deepresearch.git
cd deepresearch
```

**Last resort — nothing transferred:** rebuild from the file manifest in section 10
(every file's purpose is listed). Prefer transferring; rebuilding is error-prone.

## 4. Environment setup

Requires **Python 3.11+** (3.12 recommended). First install pulls `torch` via
`sentence-transformers` (a few hundred MB) and downloads the BGE embedding model on first
use (~130 MB) — normal, one-time.

**One-command setup** (creates the venv, installs everything, writes `.env`):
```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1   # Windows
# macOS/Linux:  bash scripts/setup.sh
```

Or do it manually:
```powershell
cd deepresearch
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements-dev.txt   # app deps + pytest + ruff
```

## 5. Configuration

```powershell
copy .env.example .env
```
Edit `.env`:
- **Demo mode (no keys):** `DEMO_MODE=1`
- **Live via Groq (free, recommended):** get a key at <https://console.groq.com/keys>, then
  `LLM_PROVIDER=groq`, `GROQ_API_KEY=...`, `DEMO_MODE=0`
- **Live fully local via Ollama:** install <https://ollama.com>, `ollama pull llama3.1`, then
  `LLM_PROVIDER=ollama`, `DEMO_MODE=0`

> `.env` is gitignored — never commit real keys. If a token was ever pasted in plaintext
> anywhere, **revoke it** and generate a new fine-grained PAT scoped to this repo.

## 6. Testing ladder (do these in order)

**6.1 Offline tests — no keys, no network, proves the whole loop:**
```powershell
pytest          # all tests should pass
ruff check .    # lint clean (CI enforces this)
```
What this covers: the end-to-end LangGraph loop (incl. the revise-once path) with faked
LLM/search/reader/index, cost tracking, RAG chunking, RunResult metrics, red-team oracle.

**6.2 Demo mode UI (no keys):**
```powershell
# .env has DEMO_MODE=1
streamlit run app/ui_streamlit.py     # http://localhost:8501
```
Click **Research** → cited report, fact-check panel (one claim flagged), cost metrics.

**6.3 One-shot live run (Groq/Ollama key, no Redis/UI):**
```powershell
# .env has DEMO_MODE=0 and a provider configured
python -m app.agents "What are the differences between HTTP/2 and HTTP/3?"
```
Prints the full RunResult JSON. First run downloads the embedding model.

**6.4 Full stack (the real architecture):** see section 7.

**6.5 Eval + red-team (needs a live provider):**
```powershell
python -m eval.run_eval     # faithfulness + citation-coverage -> eval/results.md
python -m eval.redteam      # injects falsehoods; passes if the verifier refuses them
```

## 7. Running the full stack locally

Needs Redis. Easiest is Docker for Redis only, app processes native:
```powershell
docker run -d -p 6379:6379 redis:7-alpine
# then, with venv active and .env configured, run the helper:
powershell -ExecutionPolicy Bypass -File scripts\run_local.ps1
```
That launches API (`:8000`), the arq worker, and Streamlit (`:8501`) in separate windows.
Check `http://localhost:8000/health` → `queue: true` means the worker queue is connected.

**Everything in Docker instead:**
```powershell
docker compose up --build
# UI http://localhost:8501 · API docs http://localhost:8000/docs
# fully-local models: docker compose --profile ollama up --build
```

## 8. Push to GitHub

The per-repo identity is already committed. From the repo root:
```powershell
git config user.name  "aman7251"
git config user.email "ranaamansingh7251@gmail.com"   # verify, do not change global
git push -u origin main
# when prompted: username = aman7251 ; password = your NEW fine-grained PAT
```
Create the PAT at GitHub → Settings → Developer settings → Personal access tokens →
Fine-grained → repo `deepresearch` → **Contents: Read and write**. To cache it:
```powershell
git config --global credential.helper manager
```
After pushing, confirm the **CI badge** in the README goes green (Actions tab).

## 9. Deployment (free) — ship a public demo link

**Fastest: Hugging Face Spaces in demo mode (no secrets, always-on):**
1. Create a new **Streamlit** Space and push/connect this repo.
2. The top-level `app.py` is already the Spaces entrypoint (it runs the Streamlit UI), so
   no extra wiring is needed.
3. Space → Settings → Variables: set `DEMO_MODE=1`.
4. Spaces serves the cached sample run with zero secrets.
   - Spaces reads config from the README front matter. If you keep a separate README on the
     Space, start it with:
     ```yaml
     ---
     title: DeepResearch
     sdk: streamlit
     app_file: app.py
     ---
     ```

**Live demo from your laptop (temporary public URL):**
```powershell
docker compose up --build
cloudflared tunnel --url http://localhost:8501
```

**Persistent live (always-free / cheap VPS):** Oracle Cloud Always-Free ARM VM or Hetzner;
install Docker, clone, set `.env` with `GROQ_API_KEY`, `docker compose up -d`, reverse-proxy
the Streamlit port with Caddy/Nginx for TLS.

See `DEPLOY.md` for more detail.

## 10. Loose-ends checklist (finish these)

- [ ] Transfer/clone the repo onto this laptop (section 3).
- [ ] Run `scripts/setup.ps1` (or `scripts/setup.sh`) — installs deps + writes `.env` (was blocked before).
- [ ] `pytest` green; `ruff check .` clean.
- [ ] Push `main` to GitHub; CI badge green. (Was blocked by the proxy before.)
- [ ] Add a Groq key (or Ollama) and confirm a real `python -m app.agents "..."` run.
- [ ] Run `eval/run_eval.py` and commit `eval/results.md` as evidence.
- [ ] Deploy a public demo (HF Spaces demo mode; entrypoint `app.py` already exists) and put
      the URL at the top of `README.md`.
- [ ] (Stretch) Record a 60–90s demo video; link it in the README.
- [ ] (Stretch features) SSE streaming instead of UI polling; BGE reranker after retrieval;
      Langfuse tracing (compose profile); Qdrant instead of embedded Chroma.

Already done for you (no action needed): venv/setup scripts, HF Spaces entrypoint `app.py`,
`.dockerignore`, offline test suite incl. API smoke test, CI workflow.

## 11. Troubleshooting

- **`ddgs` returns no results / rate-limited:** DuckDuckGo throttles bursts. Re-run, lower
  `MAX_RESULTS_PER_QUERY`, or set a different backend in `app/tools.py`
  (`ddgs.text(..., backend="bing,google")`). The loop degrades gracefully if search is empty.
- **Ollama + JSON mode:** the code requests `response_format={"type":"json_object"}`. If a
  local model ignores it, `_safe_json` in `app/llm.py` extracts the JSON substring; if a model
  is very weak, switch to a larger one (`ollama pull qwen2.5`) or use Groq.
- **First run is slow / downloads:** `sentence-transformers` fetches the BGE model once;
  cached afterward (Docker mounts `hf-cache`).
- **`/research` returns 503:** Redis or the arq worker isn't running. Start Redis and
  `arq app.worker.WorkerSettings`, or use `DEMO_MODE=1`.
- **API can't read runs in Docker:** the `runs` volume is shared between `api` and `worker`
  in `docker-compose.yml`; don't remove it.
- **Groq 429 (rate limit):** free tier is ~30 req/min, 1k/day. Space out runs or use
  `llama-3.1-8b-instant` (`GROQ_MODEL`).

## 12. File manifest (what each file is)

```
.env.example              env template (copy to .env)
.gitignore                excludes secrets, venv, caches, models
.dockerignore             keeps the Docker build context small
README.md                 overview, architecture, quickstart, testing
DEPLOY.md                 hosting options + push commands
HANDOFF.md                this file
LICENSE                   MIT
app.py                    Hugging Face Spaces entrypoint (runs the Streamlit UI)
Dockerfile                app image (api/worker/ui share it)
docker-compose.yml        redis + api + worker + streamlit (+ ollama/qdrant profiles)
Makefile                  setup/install/test/lint/run targets
pyproject.toml            ruff + pytest config
requirements.txt          runtime deps
requirements-dev.txt      runtime + pytest + ruff
scripts/setup.ps1         one-command setup on Windows (venv + install + .env)
scripts/setup.sh          one-command setup on macOS/Linux
scripts/run_local.ps1     launch api+worker+ui on Windows
.github/workflows/ci.yml  CI: ruff + pytest (offline)

app/__init__.py           package marker / version
app/config.py             pydantic settings from .env (provider switch, limits)
app/llm.py                OpenAI-compatible client + UsageTracker (tokens/cost/latency)
app/tools.py              web_search (ddgs) + fetch_and_extract (httpx+trafilatura)
app/rag.py                EphemeralIndex (per-run Chroma) + chunking
app/agents.py             LangGraph loop: planner/search/reader/synthesizer/critic + run_research()
app/models.py             pydantic schemas: RunResult, Claim, Source, metrics
app/store.py              JSON run store (runs/<id>.json)
app/api.py                FastAPI: POST /research, GET /runs/{id}, /health
app/worker.py             arq worker that runs the loop off the request path
app/ui_streamlit.py       UI: report, citations, flagged claims, cost panel, demo mode

eval/questions.json       eval question set
eval/run_eval.py          faithfulness + citation-coverage scoreboard
eval/redteam.py           prompt-injection suite + case_passes oracle
samples/demo_run.json     cached run powering DEMO_MODE

tests/conftest.py         FakeLLM, FakeIndex, faked tools (offline)
tests/test_agents.py      end-to-end loop incl. revision path
tests/test_api.py         FastAPI /health + 404 smoke test (no Redis)
tests/test_llm.py         usage/cost math + JSON parsing
tests/test_models.py      faithfulness/coverage + demo sample round-trip
tests/test_rag.py         chunking
tests/test_redteam.py     guardrail oracle
```

## 13. Design notes / golden rules (don't break these)

1. **Never run the multi-minute agent loop inside an HTTP request** — it runs in the arq
   worker; the API only enqueues and serves status.
2. **No secrets in git.** `.env` is ignored; keys live in env / host secret store only.
3. **Per-repo git identity** — never modify the global git config.
4. **Provider-agnostic** — switching Groq ↔ Ollama is a one-line `.env` change; keep it that way.
5. **Demo mode must always work with zero keys** so reviewers can use the public link.
