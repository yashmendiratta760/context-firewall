from fastapi import FastAPI, UploadFile, File, HTTPException
from duplicacy_detection import check_and_register

app = FastAPI()


@app.post("/check-file")
async def check_file(file: UploadFile = File(...)):
    content = await file.read()

    if not content:
        raise HTTPException(status_code=400, detail="Empty file")

    result = check_and_register(content, file.filename or "unnamed")
    return result
