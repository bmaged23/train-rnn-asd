"""Start one model-serving process. Environment variables configure assets/device."""
import os

import uvicorn

# Load the project's .env consistently with the research scripts.
import config  # noqa: F401


def main():
    uvicorn.run("src.serving.app:create_app", factory=True,
                host=os.getenv("ASD_HOST", "127.0.0.1"),
                port=int(os.getenv("ASD_PORT", "8000")), workers=1,
                limit_concurrency=16, timeout_keep_alive=5)


if __name__ == "__main__":
    main()
