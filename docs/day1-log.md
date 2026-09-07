# Day 1 Log — Environment Setup + First LLM Call
### Agentic RAG Project (rag-project-1)

---

## Goal for the day
Set up the project environment and confirm a working LangChain + Groq LLM call before moving into document loading (Day 2).

---

## Steps completed

### 1. Got a Groq API key
- Signed up at console.groq.com
- Generated an API key from API Keys → Create API Key

### 2. Created project folder + virtual environment
```powershell
mkdir rag-project-1
cd rag-project-1
python -m venv venv
venv\Scripts\activate
```

### 3. Created folder structure
```powershell
mkdir data
mkdir src
mkdir data\documents
```
Plus empty files: `src/__init__.py`, `app.py`, `config.py`, `.env.example`, `requirements.txt`, `.gitignore`

### 4. Installed packages
`requirements.txt`:
```
langchain
langchain-groq
python-dotenv
```
```powershell
pip install -r requirements.txt
```

### 5. Stored API key in `.env`
```dotenv
GROQ_API_KEY=your_key_here
```
Added to `.gitignore`:
```
venv/
.env
__pycache__/
```

### 6. Wrote and ran the first LLM call — `src/test_llm.py`
```python
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()  # reads your .env file

llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0)

response = llm.invoke("In one sentence, what is Retrieval-Augmented Generation?")
print(response.content)
```

### 7. Confirmed success
Output:
> Retrieval‑Augmented Generation is a method that first retrieves relevant external documents or facts and then feeds them into a generative model so the model can produce responses that are grounded in that retrieved knowledge.

### 8. Initialized Git and committed
```powershell
git init
git add .
git status   # confirmed .env and venv/ were NOT in the tracked list
git commit -m "Day 1: project setup + first working LLM call"
```
Result: 8 files committed (`app.py`, `config.py`, `requirements.txt`, `.gitignore`, `.env.example`, `src/__init__.py`, `src/list_models.py`, `src/test_llm.py`) — `.env` and `venv/` correctly excluded.

### 9. Recreated `venv` and `.env` after accidental local deletion
`venv/` and `.env` were deleted from disk outside of Git (see Error #7 below). Recreated locally:
```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```
Generated a fresh Groq API key and rewrote `.env`. Re-ran `test_llm.py` to confirm it still worked.

### 10. Pushed to GitHub
```powershell
git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPO-NAME.git
git branch -M main
git push -u origin main
```
Verified on GitHub: all 8 project files visible, commit message intact, **no `.env` or `venv/` present** in the remote repo.

---

## Errors faced + fixes

| # | Error | Cause | Fix |
|---|---|---|---|
| 1 | `mkdir : A positional parameter cannot be found that accepts argument 'src'` | PowerShell's `mkdir` doesn't take multiple space-separated folder names like Bash does | Run separately (`mkdir data`, `mkdir src`) or comma-separated: `mkdir data, src` |
| 2 | `ModuleNotFoundError: No module named 'dotenv'` | Packages were installed, but VS Code's Run button was using a different Python interpreter than the activated venv terminal | Ran the script directly from the terminal (`python src/test_llm.py`) instead of the Run button; reselected the venv interpreter via `Ctrl+Shift+P` → Python: Select Interpreter |
| 3 | `venv\Scripts\activate` blocked — `running scripts is disabled on this system` | Windows PowerShell execution policy blocks local scripts by default | `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` (one-time fix, user-level only) |
| 4 | `groq.NotFoundError: 404 - model 'llama-3.3-70b-versatile' does not exist` | Model name assumed from general knowledge wasn't available on this Groq account | Queried the account's actual available models directly via the Groq API |
| 5 | `groq.NotFoundError: 404 - model 'llama-3.1-8b-instant' does not exist` | Same root cause as #4 — this account's model lineup doesn't include Llama 3.x models at all | Wrote a small script (`src/list_models.py`) to call `GET /v1/models` and print the account's real available models |
| 6 | `groq.AuthenticationError: 401 - Invalid API Key` | Pasted API key had a leading space after `=` in `.env`, and the key had also been exposed in chat | Rotated the key (revoked old, created new), fixed `.env` formatting with no space: `GROQ_API_KEY=key_here` |
| 7 | `venv/` and `.env` accidentally deleted from disk | Deleted locally while cleaning up before the Git commit | Recreated `venv` from `requirements.txt`, generated a fresh Groq key and rewrote `.env` (had a backup as a safety net) |
| 8 | `Unable to copy '...venvlauncher.exe' to '...venv\Scripts\python.exe'` | Leftover locked files from the previous `venv` folder, likely still referenced by an open VS Code terminal | Closed the terminal / VS Code, force-deleted the old folder (`Remove-Item -Recurse -Force venv`), then recreated `venv` cleanly |
| 9 | `fatal: not a git repository (or any of the parent directories): .git` | Ran `git` commands before running `git init`, or from the wrong folder | Confirmed working directory with `pwd`, then ran `git init` |

**Security note:** an API key was accidentally pasted in chat during setup. It was rotated immediately. Going forward: never paste API keys into chat, only into local `.env` files.

---

## Key lessons from Day 1
- Don't assume a model name from memory/documentation — LLM provider catalogs change often. When a model 404s, query the provider's `/models` endpoint directly to get ground truth for that specific account.
- Always double-check `git status` before the first commit to confirm `.gitignore` is actually excluding `.env` and `venv/` — it worked correctly here, but it's worth verifying every time on a new repo.

---

## Confirmed working config for this project
- **LLM provider:** Groq
- **Working model:** `openai/gpt-oss-20b` (upgrade option: `openai/gpt-oss-120b`)
- **Environment:** Python venv, VS Code, PowerShell
- **Version control:** Git repo initialized, first commit pushed to GitHub with `.env` and `venv/` correctly excluded

---

## Next up: Day 2 — Document loading & text extraction
