# main.py

import json
import subprocess
import sys
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from duplicacy_detection import check_and_register
from prompt_refinement import PromptRefinementAgent


app = FastAPI(
    title="AI Context Firewall",
    description=(
        "Privacy protection, duplicate document detection, "
        "and prompt refinement through a single API endpoint."
    ),
    version="1.0.0",
)

BASE_DIR = Path(__file__).resolve().parent
PRIVACY_AGENT_PATH = BASE_DIR / "privacy_agent.py"

MAX_FILE_SIZE = 15 * 1024 * 1024

ALLOWED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".xlsx",
    ".txt",
    ".md",
    ".csv",
    ".png",
    ".jpg",
    ".jpeg",
    ".json",
    ".py",
    ".sql",
}

refinement_agent = PromptRefinementAgent()



def run_privacy_agent(
    prompt: str,
    filename: str | None = None,
    content: bytes | None = None,
) -> dict:
    """
    Run privacy_agent.py and return its JSON report.

    Supports:
    1. Prompt without a document.
    2. Prompt with one uploaded document.
    """

    # ---------------------------------------------------------
    # 1. Create a temporary workspace
    # ---------------------------------------------------------

    with tempfile.TemporaryDirectory(
        prefix="context_firewall_"
    ) as temp:
        temp_dir = Path(temp)
        prompt_path = temp_dir / "prompt.txt"
        output_dir = temp_dir / "output"

        prompt_path.write_text(prompt, encoding="utf-8")
        output_dir.mkdir()

        command = [
            sys.executable,
            str(PRIVACY_AGENT_PATH),
            "--prompt-file",
            str(prompt_path),
            "--out-dir",
            str(output_dir),
            "--no-vault",
            "--json",
        ]

        # -----------------------------------------------------
        # 2. Add attachment arguments only if a file exists
        # -----------------------------------------------------

        if filename is not None and content is not None:
            extension = Path(filename).suffix.lower()
            input_path = temp_dir / f"input{extension}"
            input_path.write_bytes(content)

            command.extend([
                "--attach",
                str(input_path),
            ])

        # -----------------------------------------------------
        # 3. Run privacy agent
        # -----------------------------------------------------

        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=180,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise HTTPException(
                status_code=504,
                detail="Privacy agent timed out.",
            ) from exc

        if result.returncode != 0:
            print(
                "Privacy agent failed:",
                result.stderr,
                file=sys.stderr,
            )
            raise HTTPException(
                status_code=500,
                detail="Privacy agent failed.",
            )

        # -----------------------------------------------------
        # 4. Parse JSON report
        # -----------------------------------------------------

        try:
            report = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise HTTPException(
                status_code=500,
                detail="Privacy agent returned invalid JSON.",
            ) from exc

        prompt_report = report.get("prompt", {})

        sanitized_prompt = prompt_report.get(
            "sanitized_prompt", ""
        )

        cloud_eligible = prompt_report.get(
            "cloud_eligible", False
        )

        # -----------------------------------------------------
        # 5. Handle prompt-only requests
        # -----------------------------------------------------

        if filename is None or content is None:
            return {
                "sanitized_prompt": sanitized_prompt,
                "cloud_eligible": cloud_eligible,
                "document_decision": None,
                "attachment_action": "no_file",
                "document_content": None,
                "privacy_report": report,
            }

        # -----------------------------------------------------
        # 6. Handle uploaded document
        # -----------------------------------------------------

        attachments = report.get("attachments", [])

        if not attachments:
            document_decision = "REMOVE"
            attachment_action = "remove"
            document_content = None

        else:
            attachment = attachments[0]

            attachment_action = str(
                attachment.get("action", "remove")
            ).lower()

            document_decision = (
                "KEEP"
                if attachment_action in {
                    "keep", "sanitize", "review"
                }
                else "REMOVE"
            )

            document_content = None

            if document_decision == "KEEP":
                if attachment_action == "sanitize":
                    sanitized_copy = attachment.get(
                        "sanitized_copy"
                    )

                    if not sanitized_copy:
                        document_decision = "REMOVE"
                    else:
                        sanitized_path = Path(
                            sanitized_copy
                        ).resolve()

                        try:
                            sanitized_path.relative_to(
                                output_dir.resolve()
                            )
                        except ValueError:
                            document_decision = "REMOVE"
                        else:
                            if not sanitized_path.is_file():
                                document_decision = "REMOVE"
                            else:
                                document_content = (
                                    sanitized_path.read_bytes()
                                )

                else:
                    document_content = content

        # -----------------------------------------------------
        # 7. Return results
        # -----------------------------------------------------

        return {
            "sanitized_prompt": sanitized_prompt,
            "cloud_eligible": cloud_eligible,
            "document_decision": document_decision,
            "attachment_action": attachment_action,
            "document_content": document_content,
            "privacy_report": report,
        }



@app.post("/context-firewall")
async def context_firewall(
    prompt: str = Form(...),
    file: UploadFile | None = File(default=None),
    local_only: bool = Form(default=True),
    external_provider_enabled: bool = Form(default=False),
    chat_id: str = Form("1")
):
    """
    Execute all three agents through one endpoint.

    1. Privacy agent: processes the prompt and optional document.
    2. Duplicate detection: runs only for approved documents.
    3. Prompt refinement: runs on the sanitized prompt.
    """

    # ---------------------------------------------------------
    # 1. Validate request
    # ---------------------------------------------------------

    if not prompt.strip():
        raise HTTPException(
            status_code=400,
            detail="Prompt cannot be empty.",
        )

    # Defaults when no file is uploaded.
    filename = None
    content = None

    # Validate and read the file only when provided.
    if file is not None:
        if not file.filename:
            raise HTTPException(
                status_code=400,
                detail="A filename is required.",
            )

        filename = Path(file.filename).name
        extension = Path(filename).suffix.lower()

        if extension not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail="Unsupported file type.",
            )

        if not PRIVACY_AGENT_PATH.is_file():
            raise HTTPException(
                status_code=500,
                detail="privacy_agent.py was not found.",
            )

        content = await file.read(MAX_FILE_SIZE + 1)

        if not content:
            raise HTTPException(
                status_code=400,
                detail="Empty files are not allowed.",
            )

        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail="File exceeds the 15 MB limit.",
            )

    # ---------------------------------------------------------
    # 2. Privacy agent
    # ---------------------------------------------------------

    try:
        privacy = run_privacy_agent(
            prompt=prompt,
            filename=filename,
            content=content,
        )
    except Exception as exc:
        print(
            "Privacy agent failed:",
            repr(exc),
            file=sys.stderr,
        )
        raise HTTPException(
            status_code=500,
            detail="Privacy agent failed.",
        ) from exc

    sanitized_prompt = privacy["sanitized_prompt"]
    document_decision = privacy.get("document_decision")
    cloud_eligible = privacy["cloud_eligible"]

    # ---------------------------------------------------------
    # 3. Duplicate detection
    # ---------------------------------------------------------

    if file is None:
        duplicate_result = {
            "checked": False,
            "reason": "No document was uploaded.",
        }

    elif document_decision == "KEEP":
        try:
            duplicate_result = check_and_register(
                privacy["document_content"],
                filename,
                chat_id
            )
        except Exception as exc:
            print(
                "Duplicate detection failed:",
                repr(exc),
                file=sys.stderr,
            )
            raise HTTPException(
                status_code=500,
                detail="Duplicate detection failed.",
            ) from exc

    else:
        duplicate_result = {
            "checked": False,
            "reason": (
                "Document was not approved by the privacy agent."
            ),
        }

    # ---------------------------------------------------------
    # 4. Prompt refinement
    # ---------------------------------------------------------

    if not cloud_eligible:
        refinement_result = {
            "refined_prompt": None,
            "status": "blocked",
            "reason": (
                "The privacy agent blocked the prompt. "
                "Prompt refinement was not run."
            ),
        }

    elif not sanitized_prompt or not sanitized_prompt.strip():
        refinement_result = {
            "refined_prompt": None,
            "status": "skipped",
            "reason": "The sanitized prompt is empty.",
        }

    else:
        try:
            refinement_result = await refinement_agent.refine(
                prompt=sanitized_prompt,
                local_only=local_only,
                external_provider_enabled=external_provider_enabled,
            )
        except Exception as exc:
            print(
                "Prompt refinement failed:",
                repr(exc),
                file=sys.stderr,
            )
            raise HTTPException(
                status_code=500,
                detail="Prompt refinement failed.",
            ) from exc

    # ---------------------------------------------------------
    # 5. Combined response
    # ---------------------------------------------------------

    return {
        "file_uploaded": file is not None,
        "privacy": {
            "sanitized_prompt": sanitized_prompt,
            "document_decision": document_decision,
            "original_attachment_action": privacy.get(
                "attachment_action"
            ),
        },
        "duplicate_detection": duplicate_result,
        "prompt_refinement": refinement_result,
    }


@app.get("/health")
def health():
    return {"status": "ok"}
