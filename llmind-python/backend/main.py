from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.jobs_router import router as jobs_router
from backend.projection.router import router as projection_router
from backend.related_projects.router import router as related_projects_router
from backend.taxonomy.router import router as taxonomy_router

app = FastAPI()

# Allow the browser to call the backend directly (bypassing the Next.js dev
# rewrite proxy, which fails to deliver responses for long-running LLM requests
# — see DESIGN-SPACE-TESTING.md). No credentials are used, so "*" is safe here.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(related_projects_router)
app.include_router(taxonomy_router)
app.include_router(projection_router)
app.include_router(jobs_router)


@app.get("/")
async def root():
    return {"message": "Hello World"}
