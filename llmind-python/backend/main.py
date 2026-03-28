from fastapi import FastAPI

from backend.related_projects.router import router as related_projects_router
from backend.taxonomy.router import router as taxonomy_router

app = FastAPI()
app.include_router(related_projects_router)
app.include_router(taxonomy_router)


@app.get("/")
async def root():
    return {"message": "Hello World"}
