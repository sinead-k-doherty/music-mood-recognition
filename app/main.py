from fastapi import FastAPI, status
import uvicorn

app = FastAPI()


@app.post("/api/test", status_code=status.HTTP_200_OK)
async def test():
    return
    
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=5001, reload=True, log_level="info")
