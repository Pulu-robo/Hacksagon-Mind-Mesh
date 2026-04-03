"""
Command Line Interface for Data Science Copilot
"""

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import print as rprint
from pathlib import Path
import json
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from orchestrator import DataScienceCopilot

app = typer.Typer(
    name="datascience-copilot",
    help="AI-powered Data Science Copilot for automated data analysis and modeling",
    add_completion=False
)

console = Console()

@app.command()
def version():
    """Show version information."""
    copilot = DataScienceCopilot()
    version_info = copilot.get_version_info()
    
    panel = Panel.fit(
        f"[bold cyan]Data Science Copilot[/bold cyan]\n"
        f"Version: [green]{version_info['version']}[/green]\n"
        f"Build: [yellow]{version_info['build']}[/yellow]\n"
        f"Release Date: [magenta]{version_info['release_date']}[/magenta]",
        title="Version Information",
        border_style="blue"
    )
    
    console.print(panel)
 @app.command()
def profile(
    file_path: str = typer.Argument(..., help="Path to dataset file"),
    output: str = typer.Option(None, "--output", "-o", help="Output file path for profiling report")
):
    """
    Profile dataset and generate insights.
    
    Example:
        python cli.py profile data.csv --output report.json
    """
    from tools.data_profiling import profile_dataset
    
    if output is None:
        output = f"./outputs/data/profile_{Path(file_path).stem}.json"
    
    console.print(f"\n📊 [bold]Profiling:[/bold] {file_path}\n")
    
    # Profile
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
        task = progress.add_task("Analyzing dataset...", total=None)
        result = profile_dataset(file_path)
        progress.update(task, completed=True)
    
    # Save report
    with open(output, "w") as f:
        json.dump(result, f, indent=4)
    
    console.print(f"\n[green]✓ Profiling complete! Report saved to: {output}[/green]") 

@app.command()
def train(
    file_path: str = typer.Argument(..., help="Path to prepared dataset"),
    target: str = typer.Argument(..., help="Target column name"),
    task_type: str = typer.Option("auto", "--task-type", help="Task type (classification/regression/auto)")
):
    """
    Train baseline models on prepared dataset.
    
    Example:
        python cli.py train cleaned_data.csv Survived --task-type classification
    """
    from tools.model_training import train_baseline_models
    
    console.print(f"\n🤖 [bold]Training Models[/bold]\n")
    console.print(f"📊 Dataset: {file_path}")
    console.print(f"🎯 Target: {target}\n")
    
    # Train
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
        task = progress.add_task("Training baseline models...", total=None)
        result = train_baseline_models(file_path, target, task_type)
        progress.update(task, completed=True)
    
    if "error" in result:
        console.print(f"[red]✗ Error: {result['message']}[/red]")
        raise typer.Exit(1)
    
    # Display results
    console.print(f"\n[green]✓ Training Complete![/green]\n")
    console.print(f"Task Type: {result['task_type']}")
    console.print(f"Features: {result['n_features']}")
    console.print(f"Samples: {result['n_samples']}\n")
    
    # Model comparison table
    table = Table(title="Model Performance")
    table.add_column("Model", style="cyan")
    
    # Add metric columns based on task type
    if result["task_type"] == "classification":
        table.add_column("Accuracy", justify="right")
        table.add_column("F1 Score", justify="right")
    else:
        table.add_column("R² Score", justify="right")
        table.add_column("RMSE", justify="right")
    
    for model_name, model_result in result["models"].items():
        if "test_metrics" in model_result:
            metrics = model_result["test_metrics"]
            if result["task_type"] == "classification":
                table.add_row(
                    model_name,
                    f"{metrics['accuracy']:.4f}",
                    f"{metrics['f1']:.4f}"
                )
            else:
                table.add_row(
                    model_name,
                    f"{metrics['r2']:.4f}",
                    f"{metrics['rmse']:.4f}"
                )
    
    console.print(table)
    
    # Best model
    console.print(f"\n🏆 [bold]Best Model:[/bold] {result['best_model']['name']}")
    console.print(f"   Score: {result['best_model']['score']:.4f}")
    console.print(f"   Path: {result['best_model']['model_path']}")


@app.command()
def cache_stats():
    """Show cache statistics."""
    copilot = DataScienceCopilot()
    stats = copilot.get_cache_stats()
    
    table = Table(title="Cache Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="white")
    
    table.add_row("Total Entries", str(stats["total_entries"]))
    table.add_row("Valid Entries", str(stats["valid_entries"]))
    table.add_row("Expired Entries", str(stats["expired_entries"]))
    table.add_row("Size", f"{stats['size_mb']} MB")
    
    console.print()
    console.print(table)


@app.command()
def clear_cache():
    """Clear all cached results."""
    copilot = DataScienceCopilot()
    copilot.clear_cache()
    console.print("[green]✓ Cache cleared successfully[/green]")


if __name__ == "__main__":
    app()
