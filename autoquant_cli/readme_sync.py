from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from smartpy.utility.log_util import getLogger

from autoquant_cli.data import workspace_root

logger = getLogger(__name__)

SETUP_REPO_URL = "https://github.com/autoquantai/autoquant-setup.git"
SETUP_REPO_BRANCH = "main"
SETUP_REPO_DIRNAME = "autoquant-setup"
README_FILES = ("README.md", "INSTALL.md", "UPDATE.md")
PIP_PACKAGE_TARGET = "git+https://github.com/autoquantai/autoquant-cli.git@main"


def setup_repo_dir() -> Path:
    return workspace_root() / SETUP_REPO_DIRNAME


def _run_git(command: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *command],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )


def _run_git_checked(command: list[str], cwd: Path | None = None, label: str = "git command") -> subprocess.CompletedProcess[str]:
    result = _run_git(command, cwd=cwd)
    if result.returncode == 0:
        return result
    stderr = (result.stderr or "").strip()
    stdout = (result.stdout or "").strip()
    details = "\n".join(part for part in [stderr, stdout] if part)
    raise RuntimeError(f"{label} failed: {' '.join(command)}\n{details}".strip())


def _rev_parse(repo_dir: Path, ref: str) -> str:
    result = _run_git_checked(["rev-parse", ref], cwd=repo_dir, label="git rev-parse")
    value = result.stdout.strip()
    if not value:
        raise RuntimeError(f"Empty git ref value for {ref}")
    return value


def ensure_setup_repo_synced(repo_url: str = SETUP_REPO_URL, branch: str = SETUP_REPO_BRANCH) -> dict[str, str]:
    repo_dir = setup_repo_dir()
    if repo_dir.exists() and not repo_dir.is_dir():
        raise RuntimeError(f"Setup repo path exists but is not a directory: {repo_dir}")
    if not repo_dir.exists():
        repo_dir.parent.mkdir(parents=True, exist_ok=True)
        clone_result = _run_git(
            ["clone", "--branch", branch, "--single-branch", repo_url, str(repo_dir)],
            cwd=repo_dir.parent,
        )
        if clone_result.returncode != 0:
            stderr = (clone_result.stderr or "").strip()
            stdout = (clone_result.stdout or "").strip()
            details = "\n".join(part for part in [stderr, stdout] if part)
            raise RuntimeError(f"Unable to clone setup repository into {repo_dir}\n{details}".strip())
    if not (repo_dir / ".git").exists():
        raise RuntimeError(f"Setup repo directory is not a git clone: {repo_dir}")
    _run_git_checked(["fetch", "origin", branch], cwd=repo_dir, label="git fetch")
    baseline_commit = _rev_parse(repo_dir, "HEAD")
    latest_commit = _rev_parse(repo_dir, f"origin/{branch}")
    return {
        "repo_dir": str(repo_dir),
        "repo_url": repo_url,
        "branch": branch,
        "baseline_commit": baseline_commit,
        "latest_commit": latest_commit,
    }


def diff_file_between_refs(repo_dir: Path, baseline_commit: str, latest_commit: str, rel_path: str) -> str:
    result = _run_git_checked(
        ["diff", "--unified=3", baseline_commit, latest_commit, "--", rel_path],
        cwd=repo_dir,
        label="git diff",
    )
    return result.stdout


def fast_forward_setup_repo(repo_dir: Path, branch: str = SETUP_REPO_BRANCH) -> str:
    _run_git_checked(["checkout", branch], cwd=repo_dir, label="git checkout")
    _run_git_checked(["merge", "--ff-only", f"origin/{branch}"], cwd=repo_dir, label="git merge --ff-only")
    return _rev_parse(repo_dir, "HEAD")


def _pip_path() -> Path:
    return workspace_root() / "venv" / "autoquant" / "bin" / "pip"


def sync_setup_repo() -> dict[str, Any]:
    sync_state = ensure_setup_repo_synced()
    repo_dir = Path(sync_state["repo_dir"])
    baseline_commit = sync_state["baseline_commit"]
    latest_commit = sync_state["latest_commit"]
    head_after_sync = baseline_commit
    updated = False
    if baseline_commit != latest_commit:
        head_after_sync = fast_forward_setup_repo(repo_dir, branch=SETUP_REPO_BRANCH)
        updated = True
    return {
        "repo_url": sync_state["repo_url"],
        "repo_dir": sync_state["repo_dir"],
        "branch": sync_state["branch"],
        "baseline_commit": baseline_commit,
        "latest_commit": latest_commit,
        "head_after_sync": head_after_sync,
        "updated": updated,
    }


def run_update() -> dict[str, Any]:
    sync_result = sync_setup_repo()
    pip_path = _pip_path()
    if not pip_path.exists():
        raise RuntimeError(f"Pip executable not found: {pip_path}")
    command = [str(pip_path), "install", "--upgrade", "--force-reinstall", PIP_PACKAGE_TARGET]
    pip_result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )
    if pip_result.returncode != 0:
        stderr = (pip_result.stderr or "").strip()
        stdout = (pip_result.stdout or "").strip()
        details = "\n".join(part for part in [stderr, stdout] if part)
        raise RuntimeError(f"Update failed for command: {' '.join(command)}\n{details}".strip())
    return {
        "command": command,
        "exit_code": pip_result.returncode,
        "stdout": pip_result.stdout,
        "stderr": pip_result.stderr,
        "setup_repo_dir": sync_result["repo_dir"],
        "setup_repo_branch": sync_result["branch"],
        "setup_repo_commit_before_update": sync_result["baseline_commit"],
        "setup_repo_latest_commit": sync_result["latest_commit"],
        "setup_repo_head_after_update": sync_result["head_after_sync"],
        "setup_repo_updated": sync_result["updated"],
    }


def get_readme_diff() -> dict[str, Any]:
    sync_state = ensure_setup_repo_synced()
    repo_dir = Path(sync_state["repo_dir"])
    baseline_commit = sync_state["baseline_commit"]
    latest_commit = sync_state["latest_commit"]
    file_diffs: list[dict[str, Any]] = []
    for rel_path in README_FILES:
        diff_text = diff_file_between_refs(repo_dir, baseline_commit, latest_commit, rel_path)
        file_diffs.append(
            {
                "path": rel_path,
                "changed": bool(diff_text.strip()),
                "diff": diff_text,
            }
        )
    return {
        "repo_url": sync_state["repo_url"],
        "repo_dir": sync_state["repo_dir"],
        "branch": sync_state["branch"],
        "baseline_commit": baseline_commit,
        "latest_commit": latest_commit,
        "has_changes": any(item["changed"] for item in file_diffs),
        "files": file_diffs,
    }
