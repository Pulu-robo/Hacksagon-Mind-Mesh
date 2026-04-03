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
def profile(
    file_path: str = typer.Argument(..., help="Path to dataset file")
):
    """
    Quick profile of a dataset (basic statistics and quality checks).
    
    Example:
        python cli.py profile data.csv
    """
    from tools.data_profiling import profile_dataset, detect_data_quality_issues
    
    console.print(f"\n📊 [bold]Profiling:[/bold] {file_path}\n")
    
    # Profile
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
        task1 = progress.add_task("Analyzing dataset...", total=None)
        profile = profile_dataset(file_path)
        progress.update(task1, completed=True)
    
    # Display basic info
    info_table = Table(title="Dataset Information", show_header=False)
    info_table.add_column("Property", style="cyan")
    info_table.add_column("Value", style="white")
    
    info_table.add_row("Rows", str(profile["shape"]["rows"]))
    info_table.add_row("Columns", str(profile["shape"]["columns"]))
    info_table.add_row("Memory", f"{profile['memory_usage']['total_mb']} MB")
    info_table.add_row("Null %", f"{profile['overall_stats']['null_percentage']}%")
    info_table.add_row("Duplicates", str(profile['overall_stats']['duplicate_rows']))
    
    console.print()
    console.print(info_table)
    
    # Column types
    console.print("\n[bold]Column Types:[/bold]")
    console.print(f"  Numeric: {len(profile['column_types']['numeric'])}")
    console.print(f"  Categorical: {len(profile['column_types']['categorical'])}")
    console.print(f"  Datetime: {len(profile['column_types']['datetime'])}")
    
    # Detect issues
    console.print("\n[bold]Quality Check:[/bold]")
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
        task2 = progress.add_task("Detecting quality issues...", total=None)
        issues = detect_data_quality_issues(file_path)
        progress.update(task2, completed=True)
    
    console.print(f"  🔴 Critical: {issues['summary']['critical_count']}")
    console.print(f"  🟡 Warnings: {issues['summary']['warning_count']}")
    console.print(f"  🔵 Info: {issues['summary']['info_count']}")


@app.command()
def clean(
    file_path: str = typer.Argument(..., help="Path to dataset file"),
    output: str = typer.Option(None, "--output", "-o", help="Output file path"),
    strategy: str = typer.Option("auto", "--strategy", "-s", help="Cleaning strategy (auto/median/mean/mode/drop)")
):
    """
    Clean dataset (handle missing values and outliers).
    
    Example:
        python cli.py clean data.csv --output cleaned_data.csv
    """
    from tools.data_cleaning import clean_missing_values
    from tools.data_profiling import profile_dataset
    
    if output is None:
        output = f"./outputs/data/cleaned_{Path(file_path).name}"
    
    console.print(f"\n🧹 [bold]Cleaning:[/bold] {file_path}\n")
    
    # Get columns with missing values
    profile = profile_dataset(file_path)
    cols_with_nulls = {
        col: "auto"
        for col, info in profile["columns"].items()
        if info["null_count"] > 0
    }
    
    if not cols_with_nulls:
        console.print("[green]✓ No missing values found - dataset is clean![/green]")
        return
    
    console.print(f"Found {len(cols_with_nulls)} columns with missing values")
    
    # Clean
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
        task = progress.add_task("Cleaning dataset...", total=None)
        result = clean_missing_values(file_path, cols_with_nulls, output)
        progress.update(task, completed=True)
    
    console.print(f"\n[green]✓ Cleaned dataset saved to: {output}[/green]")
    console.print(f"  Rows: {result['original_rows']} → {result['final_rows']}")


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
