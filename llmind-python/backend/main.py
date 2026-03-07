from fastapi import FastAPI

from backend.related_projects.router import router as related_projects_router

app = FastAPI()
app.include_router(related_projects_router)


@app.get("/")
async def root():
    return {"message": "Hello World"}
