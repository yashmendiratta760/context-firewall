from fastapi import FastAPI, UploadFile, File, HTTPException
from duplicacy_detection import check_and_register
from pydantic import BaseModel, Field
from prompt_refinement import PromptRefinementAgent

refinement_agent = PromptRefinementAgent()


app = FastAPI()


@app.post("/check-file")
async def check_file(file: UploadFile = File(...)):
    content = await file.read()

    if not content:
        raise HTTPException(status_code=400, detail="Empty file")

    result = check_and_register(content, file.filename or "unnamed")
    return result



class RefinementRequest(BaseModel):
    prompt: str = Field(..., max_length=10000)
    local_only: bool = True
    external_provider_enabled: bool = False


@app.post("/refine-prompt")
async def refine_prompt(request: RefinementRequest):
    return await refinement_agent.refine(
        prompt=request.prompt,
        local_only=request.local_only,
        external_provider_enabled=request.external_provider_enabled,
    )
