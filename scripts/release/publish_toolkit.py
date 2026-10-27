"""Publish the tested checkpoint-required toolkit and verify an anonymous pull."""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path


def run(*args, env=None, capture=False):
    return subprocess.run(args, check=True, env=env, text=True,
                          stdout=subprocess.PIPE if capture else None, timeout=300)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default="asd-window-toolkit:ci")
    args = parser.parse_args()
    revision = os.environ.get("GITHUB_SHA", "")
    token = os.environ.get("DOCKERHUB_TOKEN", "")
    if not re.fullmatch(r"[0-9a-f]{40}", revision) or not token:
        raise SystemExit("A full GITHUB_SHA and DOCKERHUB_TOKEN are required")
    repository = "bmaged23/train-rnn-asd"
    tag = f"{repository}:sha-{revision}"
    with tempfile.TemporaryDirectory(prefix="asd-publish-auth-") as auth, tempfile.TemporaryDirectory(prefix="asd-anonymous-") as anonymous:
        authenticated, public = dict(os.environ), dict(os.environ)
        authenticated["DOCKER_CONFIG"], public["DOCKER_CONFIG"] = auth, anonymous
        public.pop("DOCKERHUB_TOKEN", None)
        # The secret enters docker through stdin, never command arguments or logs.
        subprocess.run(["docker", "login", "--username", "bmaged23", "--password-stdin"],
                       input=token, text=True, check=True, env=authenticated, timeout=30,
                       stdout=subprocess.DEVNULL)
        run("docker", "tag", args.image, tag)
        run("docker", "push", tag, env=authenticated)
        digests = json.loads(run("docker", "image", "inspect", tag, "--format", "{{json .RepoDigests}}", capture=True).stdout)
        immutable = next(value for value in digests if value.startswith(repository + "@sha256:"))
        run("docker", "pull", immutable, env=public)
        for module in ("scripts.validation.validate_inference_container", "scripts.validation.validate_api_container"):
            run(sys.executable, "-m", module, "--image", immutable, env=public)
        # Promote the moving tag only after the public immutable image passes.
        run("docker", "tag", immutable, repository + ":cpu")
        run("docker", "push", repository + ":cpu", env=authenticated)
        run("docker", "pull", repository + ":cpu", env=public)
        promoted = json.loads(run("docker", "image", "inspect", repository + ":cpu", "--format", "{{json .RepoDigests}}", capture=True).stdout)
        if immutable not in promoted:
            raise RuntimeError("Published cpu tag does not match the evaluated digest")
    evidence = dict(source_revision=revision, image=immutable, tag=repository + ":cpu",
                    scope="CPU toolkit; synthetic contract verification; trained checkpoint not included")
    output = Path(os.getenv("RUNNER_TEMP", "/tmp")) / "asd-release-evidence.json"
    output.write_text(json.dumps(evidence, indent=2) + "\n")
    if os.getenv("GITHUB_STEP_SUMMARY"):
        with Path(os.environ["GITHUB_STEP_SUMMARY"]).open("a") as stream:
            stream.write("## Verified CPU toolkit delivery\n\n```json\n" + json.dumps(evidence, indent=2) + "\n```\n")
    print(json.dumps(evidence))


if __name__ == "__main__":
    main()
