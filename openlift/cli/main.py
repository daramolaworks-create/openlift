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


# ... (existing imports)
from openlift.core.design import GeoMatcher, PowerAnalysis
from openlift.core.io import load_data
import pandas as pd

design_app = typer.Typer(help="Design assistant tools.")
app.add_typer(design_app, name="design")

@design_app.command("match")
def match_geos(
    data: str = typer.Option(..., help="Path to data CSV"),
    target: str = typer.Option(..., help="Target geo to match controls for"),
    date_col: str = typer.Option("date", help="Date column name"),
    geo_col: str = typer.Option("geo", help="Geo column name"),
    outcome_col: str = typer.Option("outcome", help="Outcome column name"),
    lookback: int = typer.Option(90, help="Days to look back for matching"),
    top_k: int = typer.Option(5, help="Number of controls to find")
):
    """
    Find the best matching control markets for a target market.
    """
    typer.echo(f"Loading data from {data}...")
    df = load_data(data, date_col, geo_col, outcome_col)
    
    matcher = GeoMatcher(df, date_col, geo_col, outcome_col)
    matches = matcher.find_controls(target, lookback_days=lookback, n_controls=top_k)
    
    typer.echo(f"Top {top_k} matches for {target}:")
    for rank, (geo, score) in enumerate(matches, 1):
        typer.echo(f"{rank}. {geo} (Score: {score:.4f})")

@design_app.command("power")
def power_analysis(
    data: str = typer.Option(..., help="Path to data CSV"),
    target: str = typer.Option(..., help="Target geo"),
    controls: str = typer.Option(..., help="Comma-separated list of control geos"),
    lift: float = typer.Option(0.10, help="Expected lift (e.g. 0.10 for 10%)"),
    duration: int = typer.Option(30, help="Test duration in days"),
    date_col: str = typer.Option("date", help="Date column name"),
    geo_col: str = typer.Option("geo", help="Geo column name"),
    outcome_col: str = typer.Option("outcome", help="Outcome column name"),
    simulations: int = typer.Option(20, help="Number of simulations")
):
    """
    Run power analysis to estimate detection probability.
    """
    control_list = [c.strip() for c in controls.split(",")]
    
    typer.echo(f"Loading data from {data}...")
    df = load_data(data, date_col, geo_col, outcome_col)
    
    pa = PowerAnalysis(df, date_col, geo_col, outcome_col)
    
    typer.echo(f"Simulating power for {target} with {len(control_list)} controls...")
    typer.echo(f"Effect size: {lift*100}%, Duration: {duration} days")
    
    result = pa.simulate_power(
        target, 
        control_list, 
        effect_size_pct=lift,
        test_duration_days=duration,
        simulations=simulations
    )
    
    power = result["power"]
    typer.echo(f"Estimated Power: {power*100:.1f}%")
    if power >= 0.8:
        typer.echo("✅ Sufficient power (>80%)")
    else:
        typer.echo("⚠️ Low power (<80%). Consider increasing duration or effect size.")

if __name__ == "__main__":
    app()
