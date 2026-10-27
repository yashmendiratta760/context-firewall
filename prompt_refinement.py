
import os
import re
import unicodedata
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()


class PrivacyVerificationAgent:
    """
    Basic detection for email addresses, API tokens,
    passwords, and secret keys.
    """

    PATTERNS = [
        re.compile(
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
        ),
        re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
        re.compile(
            r"\b(?:password|secret[\s_-]*key|api[\s_-]*key)"
            r"\s*[:=]\s*\S+",
            re.IGNORECASE,
        ),
    ]

    def verify(self, text: str) -> bool:
        if not isinstance(text, str):
            return False

        return not any(
            pattern.search(text) for pattern in self.PATTERNS
        )


def refine_locally(prompt: str) -> str:
    if not isinstance(prompt, str):
        return ""

    prompt = prompt.replace("\r\n", "\n").replace("\r", "\n")

    # 1. Remove non-printable control characters
    prompt = "".join(
        ch for ch in prompt
        if ch in "\n\t" or ch.isprintable()
    )

    # 2. Normalize Unicode characters
    prompt = unicodedata.normalize("NFKC", prompt)

    # 3. Normalize spaces and tabs, preserving paragraphs
    lines = [
        re.sub(r"[ \t]+", " ", line).strip()
        for line in prompt.split("\n")
    ]
    prompt = "\n".join(lines)

    # 4. Remove excessive blank lines
    prompt = re.sub(r"\n{3,}", "\n\n", prompt)

    # 5. Collapse repeated punctuation
    prompt = re.sub(r"[!]{2,}", "!", prompt)
    prompt = re.sub(r"[?]{2,}", "?", prompt)
    prompt = re.sub(r"[.]{4,}", "...", prompt)
    prompt = re.sub(r"[,;]{2,}", ",", prompt)
    prompt = re.sub(r"[-_]{3,}", "-", prompt)
    prompt = re.sub(r"[~^]{2,}", "", prompt)

    # 6. Remove punctuation-only lines
    lines = [
        line for line in prompt.split("\n")
        if not re.fullmatch(r"[^\w\s]+", line)
    ]
    prompt = "\n".join(lines)

    # 7. Remove stray punctuation between ordinary words
    prompt = re.sub(
        r"(?<=\w)\s*[,;]\s*(?=\w)",
        " ",
        prompt,
    )

    # 8. Remove isolated decorative symbols between spaces.
    # Do not remove symbols commonly used in technical text.
    prompt = re.sub(
        r"(?<=\s)[|~^]+(?=\s)",
        " ",
        prompt,
    )

    # 9. Clean spaces before common punctuation
    prompt = re.sub(r"[ \t]+([,.!?;:])", r"\1", prompt)

    # 10. Normalize spaces after punctuation
    prompt = re.sub(r"([,.!?;:])(?=\S)", r"\1 ", prompt)

    # 11. Final whitespace cleanup
    return prompt.strip()


class PromptRefinementAgent:
    def __init__(self, max_length: int = 10000):
        self.max_length = max_length
        self.verifier = PrivacyVerificationAgent()

        # Groq is optional. Local mode works without an API key.
        self.model = os.getenv(
            "GROQ_MODEL",
            "qwen/qwen3.8-27b",
        )

        api_key = os.getenv("GROQ_API_KEY")
        self.client = None

        if api_key:
            self.client = ChatGroq(
                model=self.model,
                api_key=api_key,
                temperature=0.0,
                max_retries=2,
            )

    async def refine(
        self,
        prompt: str,
        local_only: bool = True,
        external_provider_enabled: bool = False,
    ) -> dict:

        # 1. Validate input.
        if not isinstance(prompt, str) or not prompt.strip():
            return {
                "status": "error",
                "message": "Prompt cannot be empty.",
                "refined_prompt": None,
                "refinement_occurred": False,
            }

        if len(prompt) > self.max_length:
            return {
                "status": "error",
                "message": "Prompt exceeds the maximum length.",
                "refined_prompt": None,
                "refinement_occurred": False,
            }

        original = prompt.strip()

        # 2. Local-only mode: no Groq call or external transmission.
        if local_only:
            refined = refine_locally(original)

            return {
                "status": "success",
                "message": "Prompt cleaned locally using Python rules.",
                "original_prompt": original,
                "refined_prompt": refined,
                "refinement_occurred": refined != original,
                "mode": "local",
            }

        # 3. Cloud mode requires explicit permission.
        if not external_provider_enabled:
            return {
                "status": "blocked",
                "message": "External provider permission is required.",
                "refined_prompt": None,
                "refinement_occurred": False,
            }

        if self.client is None:
            return {
                "status": "error",
                "message": "Groq API key is not configured.",
                "refined_prompt": None,
                "refinement_occurred": False,
            }

        # 4. Verify before sending the prompt to Groq.
        if not self.verifier.verify(original):
            return {
                "status": "blocked",
                "message": "Input failed privacy verification.",
                "refined_prompt": None,
                "refinement_occurred": False,
            }

        # Recognize privacy placeholders such as [EMAIL].
        placeholder_pattern = r"\[[A-Z][A-Z0-9_]*\]"
        placeholders = set(
            re.findall(placeholder_pattern, original)
        )

        try:
            response = await self.client.ainvoke(
                [
                    (
                        "system",
                        """
                        You are a prompt refinement assistant.

                        Rewrite the user's input to improve clarity, grammar,
                        structure, and actionability while preserving the original
                        intent, requirements, and constraints.

                        Always try to reduce the tokens compared to the original prompt if possible,
                        Try to keep the tokens reduced to save the money.

                        Do not invent facts or execute the underlying request.
                        Preserve all privacy placeholders exactly as written.

                        Return only the refined prompt.
                        """.strip(),
                    ),
                    ("human", original),
                ]
            )

            refined = response.content

            if not isinstance(refined, str) or not refined.strip():
                raise ValueError("Empty model response.")

            refined = refined.strip()

            output_placeholders = set(
                re.findall(placeholder_pattern, refined)
            )

            if not placeholders.issubset(output_placeholders):
                return {
                    "status": "blocked",
                    "message": "Privacy placeholders were lost.",
                    "refined_prompt": None,
                    "refinement_occurred": False,
                }

            if not self.verifier.verify(refined):
                return {
                    "status": "blocked",
                    "message": "Output failed privacy verification.",
                    "refined_prompt": None,
                    "refinement_occurred": False,
                }

            return {
                "status": "success",
                "message": "Prompt refined using Groq.",
                "original_prompt": original,
                "refined_prompt": refined,
                "refinement_occurred": refined != original,
                "mode": "cloud",
                "model": self.model,
            }

        except Exception as exc:
            # Never log the prompt or API key.
            print(f"Groq refinement error: {type(exc).__name__}")

            return {
                "status": "error",
                "message": (
                    f"Prompt refinement failed: {type(exc).__name__}"
                ),
                "refined_prompt": None,
                "refinement_occurred": False,
            }
