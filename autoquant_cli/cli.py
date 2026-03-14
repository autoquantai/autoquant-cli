from __future__ import annotations

import json
from typing import Annotated, Literal

import typer

from autoquant_cli.api_client import get_openapi_json, post_json
from autoquant_cli.health import health
from autoquant_cli.readme_sync import get_readme_diff, run_update
from autoquant_cli.run_model import run_model
from autoquant_cli.validate_model import validate_model

app = typer.Typer(no_args_is_help=True, help="AutoQuant CLI.")


def _print(payload: object) -> None:
    typer.echo(json.dumps(payload, ensure_ascii=True))


def _print_pretty(payload: object) -> None:
    typer.echo(json.dumps(payload, ensure_ascii=True, indent=2))


@app.command("validate-model")
def validate_model_command(
    model_path: Annotated[str, typer.Option(...)],
    task: Annotated[Literal["classification", "regression"], typer.Option(...)],
    refresh_data: Annotated[bool, typer.Option()] = False,
) -> None:
    _print(validate_model(model_path=model_path, task=task, refresh_data=refresh_data))


@app.command("run-model")
def run_model_command(
    run_id: Annotated[str, typer.Option(...)],
    name: Annotated[str, typer.Option(...)],
    generation: Annotated[int, typer.Option(...)],
    model_path: Annotated[str, typer.Option(...)],
    log: Annotated[str, typer.Option(...)],
    parent_id: Annotated[str, typer.Option()] = "",
    reasoning: Annotated[str, typer.Option()] = "",
    task: Annotated[str, typer.Option()] = "",
) -> None:
    _print(
        run_model(
            run_id=run_id,
            name=name,
            generation=generation,
            model_path=model_path,
            log=log,
            parent_id=parent_id or None,
            reasoning=reasoning,
            task=task or None,
        )
    )


@app.command("api")
def api_command(
    path: Annotated[str, typer.Argument(...)],
    payload_json: Annotated[str, typer.Argument()] = "{}",
) -> None:
    try:
        payload = json.loads(payload_json)
    except json.JSONDecodeError as exc:
        raise typer.BadParameter(f"Invalid JSON payload: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise typer.BadParameter("Payload JSON must decode to an object")
    _print(post_json(path, payload))


@app.command("health")
def health_command() -> None:
    _print(health())


@app.command("get-openapi")
def get_openapi_command() -> None:
    _print_pretty(get_openapi_json())


@app.command("get-readme-diff")
def get_readme_diff_command() -> None:
    _print(get_readme_diff())


@app.command("run-update")
def run_update_command() -> None:
    _print(run_update())


def main() -> None:
    app()


if __name__ == "__main__":
    main()
