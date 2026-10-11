"""
main.py
--------
FastAPI application entrypoint for the Adversarial Robustness Framework for
Language-Conditioned Autonomous Driving Planners.

Serves:
  - the JSON API under /api/*  (backend/api/routes.py)
  - the built React dashboard at /  (frontend/dist, produced by `npm run build`)

Development workflow:
    Terminal 1:  uvicorn backend.main:app --reload        (API on :8000)
    Terminal 2:  cd frontend && npm run dev                (React on :5173, proxies /api -> :8000)

Production / single-process workflow:
    cd frontend && npm run build                            (writes frontend/dist)
    uvicorn backend.main:app                                 (serves the built SPA + API together on :8000)
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

from backend.api.routes import router as api_router
from backend.utils.config import FRONTEND_DIR
from backend.utils.logger import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title="Adversarial Robustness Framework for Language-Conditioned Autonomous Driving Planners",
    description=(
        "A verification layer that inspects passenger prompts, camera frames and radar sweeps "
        "for adversarial manipulation, ambiguity and sensor anomalies BEFORE a driving planner "
        "is allowed to trust them. This project does not implement a driving planner itself."
    ),
    version="2.0.0",
)

# CORS left permissive so the React dev server (Vite, :5173) can call the API
# directly during development in addition to the built-in Vite proxy.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)

DIST_DIR = FRONTEND_DIR / "dist"


@app.get("/health")
def health_check():
    """Simple liveness probe."""
    return {"status": "ok", "service": "adversarial-robustness-driving-planner"}


if DIST_DIR.exists():
    # Production mode: the React app has been built (frontend/dist exists).
    # Serve its hashed static assets, then fall back to index.html for any
    # other route so client-side navigation (e.g. the /report view) works
    # on a hard refresh too.
    app.mount("/assets", StaticFiles(directory=str(DIST_DIR / "assets")), name="assets")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        return FileResponse(str(DIST_DIR / "index.html"))
else:
    logger.warning(
        "frontend/dist not found -- React app has not been built yet. "
        "Run `cd frontend && npm run build` for single-server mode, or "
        "`npm run dev` and open http://127.0.0.1:5173 during development."
    )

    @app.get("/")
    async def frontend_not_built():
        return JSONResponse(
            status_code=200,
            content={
                "message": "API is running, but the React frontend has not been built yet.",
                "dev_instructions": "cd frontend && npm run dev  (opens on http://127.0.0.1:5173)",
                "prod_instructions": "cd frontend && npm run build  (then reload this page)",
                "docs": "/docs",
            },
        )


@app.on_event("startup")
async def warm_up_models():
    """Pre-load the prompt-detector SVM at startup so the first request isn't slow."""
    try:
        from backend.robustness.instruction_analyzer import get_instruction_analyzer
        get_instruction_analyzer()
        logger.info("Startup warm-up complete: instruction analyzer ready.")
    except Exception as exc:
        logger.warning(f"Startup warm-up failed (non-fatal, will retry lazily): {exc}")
