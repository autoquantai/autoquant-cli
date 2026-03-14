from __future__ import annotations

import json
from typing import Annotated, Literal

import typer

from autoquant_cli.commands.api_client import get_openapi_json, post_json
from autoquant_cli.commands.create_experiment import create_experiment
from autoquant_cli.commands.health import health
from autoquant_cli.commands.run_model import run_model
from autoquant_cli.commands.validate_model import validate_model

app = typer.Typer(no_args_is_help=True, help="AutoQuant CLI.")


def _print(payload: object) -> None:
    typer.echo(json.dumps(payload, ensure_ascii=True))


def _print_pretty(payload: object) -> None:
    typer.echo(json.dumps(payload, ensure_ascii=True, indent=2))


@app.command("create-experiment")
def create_experiment_command(
    name: Annotated[str, typer.Option(...)],
    from_date: Annotated[str, typer.Option(...)],
    to_date: Annotated[str, typer.Option(...)],
    task: Annotated[Literal["classification", "regression"], typer.Option(...)],
    input_ohlc_tickers: Annotated[list[str], typer.Option()] = [],
    target_ticker: Annotated[str, typer.Option(...)] = "",
    data_provider: Annotated[Literal["massive", "ccxt"], typer.Option()] = "massive",
    ccxt_exchange: Annotated[str, typer.Option()] = "",
    max_experiments: Annotated[int, typer.Option()] = 8,
    train_time_limit_minutes: Annotated[int, typer.Option()] = 30,
    refresh_data: Annotated[bool, typer.Option()] = False,
) -> None:
    _print(
        create_experiment(
            name=name,
            input_ohlc_tickers=input_ohlc_tickers,
            target_ticker=target_ticker,
            data_provider=data_provider,
            ccxt_exchange=ccxt_exchange or None,
            from_date=from_date,
            to_date=to_date,
            task=task,
            max_experiments=max_experiments,
            train_time_limit_minutes=train_time_limit_minutes,
            refresh_data=refresh_data,
        )
    )


@app.command("validate-model")
def validate_model_command(
    run_id: Annotated[str, typer.Option(...)],
    model_path: Annotated[str, typer.Option(...)],
    task: Annotated[str, typer.Option()] = "",
    refresh_data: Annotated[bool, typer.Option()] = False,
) -> None:
    _print(validate_model(run_id=run_id, model_path=model_path, task=task or None, refresh_data=refresh_data))


@app.command("run-model")
def run_model_command(
    run_id: Annotated[str, typer.Option(...)],
    name: Annotated[str, typer.Option(...)],
    generation: Annotated[int, typer.Option(...)],
    model_path: Annotated[str, typer.Option(...)],
    log: Annotated[str, typer.Option(...)],
    parent_ids: Annotated[list[str] | None, typer.Option()] = None,
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
            parent_ids=parent_ids or None,
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


def main() -> None:
    app()


if __name__ == "__main__":
    main()
