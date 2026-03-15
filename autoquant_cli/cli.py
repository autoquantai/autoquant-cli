from __future__ import annotations

import json
from typing import Annotated, Literal

import typer

from autoquant_cli.commands.api_client import post_json
from autoquant_cli.commands.create_experiment import create_experiment
from autoquant_cli.commands.health import health
from autoquant_cli.commands.run_model import run_model
from autoquant_cli.commands.validate_model import validate_model

app = typer.Typer(no_args_is_help=True, help="AutoQuant CLI.")


def _print(payload: object) -> None:
    typer.echo(json.dumps(payload, ensure_ascii=True))


@app.command("create-experiment", help="Create a new run and prepare local dataset files for model iteration.")
def create_experiment_command(
    name: Annotated[str, typer.Option(..., help="Human-readable run name used to track this experiment.")],
    from_date: Annotated[str, typer.Option(..., help="Inclusive start date for market data (YYYY-MM-DD).")],
    to_date: Annotated[str, typer.Option(..., help="Inclusive end date for market data (YYYY-MM-DD).")],
    task: Annotated[
        Literal["classification", "regression"],
        typer.Option(..., help="Learning objective for model training and evaluation."),
    ],
    input_ohlc_tickers: Annotated[
        list[str],
        typer.Option(help="Input feature tickers. Repeat flag for multiple values."),
    ] = [],
    target_ticker: Annotated[
        str,
        typer.Option(..., help="Target ticker the model predicts."),
    ] = "",
    data_provider: Annotated[
        Literal["massive", "ccxt"],
        typer.Option(help="Market data source to use for this run."),
    ] = "massive",
    ccxt_exchange: Annotated[
        str,
        typer.Option(help="Exchange name when data-provider is ccxt."),
    ] = "",
    max_experiments: Annotated[
        int,
        typer.Option(help="Maximum number of model experiments to execute for this run."),
    ] = 8,
    train_time_limit_minutes: Annotated[
        int,
        typer.Option(help="Per-model training budget in minutes."),
    ] = 30,
    refresh_data: Annotated[
        bool,
        typer.Option(help="Force refresh of local cached run data."),
    ] = False,
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


@app.command("validate-model", help="Run fast sandbox validation on a model file against run data.")
def validate_model_command(
    run_id: Annotated[str, typer.Option(..., help="Run identifier used to load data and metadata.")],
    model_path: Annotated[str, typer.Option(..., help="Path to the model Python file to validate.")],
    task: Annotated[
        str,
        typer.Option(help="Optional task override when not inferable from run metadata."),
    ] = "",
    refresh_data: Annotated[
        bool,
        typer.Option(help="Force refresh of local cached run data before validation."),
    ] = False,
) -> None:
    _print(validate_model(run_id=run_id, model_path=model_path, task=task or None, refresh_data=refresh_data))


@app.command("run-model", help="Execute full training/search for one model candidate and persist results.")
def run_model_command(
    run_id: Annotated[str, typer.Option(..., help="Run identifier this model belongs to.")],
    name: Annotated[str, typer.Option(..., help="Model experiment name within the generation.")],
    generation: Annotated[int, typer.Option(..., help="Generation number for this model experiment.")],
    model_path: Annotated[str, typer.Option(..., help="Path to the model Python file to execute.")],
    log: Annotated[str, typer.Option(..., help="Execution notes or summary for traceability.")],
    parent_ids: Annotated[
        list[str] | None,
        typer.Option(help="Optional parent experiment ids. Repeat flag for multiple values."),
    ] = None,
    reasoning: Annotated[
        str,
        typer.Option(help="Why this model was chosen and what hypothesis it tests."),
    ] = "",
    task: Annotated[
        str,
        typer.Option(help="Optional task override when not inferable from run metadata."),
    ] = "",
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


@app.command("api", help="Call a backend API endpoint with a JSON object payload.")
def api_command(
    path: Annotated[
        str,
        typer.Argument(..., help="Endpoint path, for example /run/get"),
    ],
    payload_json: Annotated[
        str,
        typer.Argument(help="JSON object payload sent in the POST body."),
    ] = "{}",
) -> None:
    try:
        payload = json.loads(payload_json)
    except json.JSONDecodeError as exc:
        raise typer.BadParameter(f"Invalid JSON payload: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise typer.BadParameter("Payload JSON must decode to an object")
    _print(post_json(path, payload))


@app.command("health", help="Check CLI/backend connectivity and return service health state.")
def health_command() -> None:
    _print(health())


def main() -> None:
    app()


if __name__ == "__main__":
    main()
