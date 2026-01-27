import typer
import json
from openlift.core.pipeline import run_geo_lift
from openlift.core.report import print_summary

app = typer.Typer(help="OpenLift: Marketing incrementality measurement.")

@app.command()
def init():
    """
    Initialize a new experiment with a template config.
    """
    template = """experiment:
  name: example_experiment
  test_geo: Lagos
  control_geos: [Ibadan, Abeokuta, Benin]
  pre_period:
    start_date: 2024-10-01
    end_date: 2024-12-31
  post_period:
    start_date: 2025-01-01
    end_date: 2025-01-21
data:
  path: data.csv
  date_col: date
  geo_col: geo
  outcome_col: outcome
"""
    with open("experiment.yaml", "w") as f:
        f.write(template)
    
    typer.echo("Created experiment.yaml template.")
    typer.echo("Next steps:")
    typer.echo("1. Edit experiment.yaml to match your data.")
    typer.echo("2. Ensure your data CSV is available.")
    typer.echo("3. Run: openlift run experiment.yaml --out results.json")

@app.command()
def run(
    config: str = typer.Argument(..., help="Path to experiment.yaml"),
    out: str = typer.Option("results.json", help="Output path for JSON results")
):
    """
    Run the geo-lift experiment.
    """
    typer.echo(f"Running experiment from {config}...")
    try:
        results = run_geo_lift(config)
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)
    
    # Save results
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    
    typer.echo(f"Results saved to {out}")
    
    # Print summary
    print_summary(results)

@app.command()
def report(
    results_path: str = typer.Argument(..., help="Path to results.json")
):
    """
    Print a human-readable report from a results file.
    """
    try:
        with open(results_path, "r") as f:
            results = json.load(f)
        print_summary(results)
    except FileNotFoundError:
        typer.echo(f"File not found: {results_path}", err=True)
        raise typer.Exit(code=1)
    except Exception as e:
        typer.echo(f"Error reading report: {e}", err=True)
        raise typer.Exit(code=1)

if __name__ == "__main__":
    app()
