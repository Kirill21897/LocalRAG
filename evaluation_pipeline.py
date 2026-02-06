import asyncio
import aiohttp
import time
import pandas as pd
from typing import List, Dict, Any, Callable
from pathlib import Path
import logging
import nest_asyncio
import json
from datetime import datetime

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

# Rich imports
from rich.console import Console
from rich.progress import (
    Progress, 
    SpinnerColumn, 
    TextColumn, 
    BarColumn, 
    TaskProgressColumn, 
    TimeElapsedColumn,
    MofNCompleteColumn
)
from rich.table import Table
from rich.panel import Panel
from rich.logging import RichHandler
from rich.text import Text
from rich.layout import Layout
from rich.syntax import Syntax
from rich import box
from rich.markdown import Markdown

# Local imports
from src.chunkers.recursive_chunker import chunk_recursive
from src.chunkers.token_chunker import chunk_token
from src.chunkers.markdown_chunker import chunk_markdown
from src.chunkers.sentence_window_chunker import chunk_sentence_window
from src.chunkers.semantic_chunker import chunk_semantic

from src.embedder import vectorize_and_upload
from src.retrieval import retrieve
from src.reranker import rerank
from src.evaluation.ragas_eval import RagasEvaluator
from config.config import PROCESSED_DATA_DIR
import config.config as cfg

# Logging setup with Rich
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, show_path=False)]
)
logger = logging.getLogger("rich")
console = Console()

class EvaluationReporter:
    """
    Handles all visualization and reporting logic for the evaluation pipeline.
    Separates presentation from logic.
    """
    def __init__(self, console: Console):
        self.console = console
        self.method_stats = {}

    def header(self):
        grid = Table.grid(expand=True)
        grid.add_column(justify="center", ratio=1)
        grid.add_row(
            Panel(
                Text("LocalRAG Evaluation Pipeline", justify="center", style="bold white on blue"),
                subtitle="v3.0 Debug & Refactored",
                style="blue"
            )
        )
        self.console.print(grid)
        self.console.print(f"[dim]Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/dim]\n")

    def section(self, title: str, color: str = "cyan"):
        self.console.rule(f"[{color}]{title}[/{color}]")

    def log_chunking_stats(self, method_name: str, chunks: List[str]):
        """Displays detailed stats about the chunking process."""
        num_chunks = len(chunks)
        avg_len = sum(len(c) for c in chunks) / num_chunks if num_chunks > 0 else 0
        
        table = Table(show_header=True, header_style="bold magenta", box=box.SIMPLE)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Total Chunks", str(num_chunks))
        table.add_row("Avg Chunk Size (chars)", f"{avg_len:.1f}")
        table.add_row("Sample Chunk (Preview)", chunks[0][:100].replace("\n", " ") + "..." if chunks else "N/A")
        
        self.console.print(Panel(table, title=f"[bold]{method_name} - Chunking Stats[/bold]", border_style="cyan"))

    def log_interaction_sample(self, method_name: str, question: str, context_details: List[Dict[str, Any]], answer: str, ground_truth: str):
        """
        Displays a detailed debug view of a single Q&A interaction.
        This helps the user see EXACTLY what the model is doing.
        """
        grid = Table.grid(expand=True)
        grid.add_column(ratio=1)
        
        # Question
        grid.add_row(Panel(Text(f"{question}", style="bold yellow"), title="Question", border_style="yellow"))
        
        # Retrieval Candidates (Minimalist View)
        cand_table = Table(box=box.SIMPLE_HEAD, show_edge=False, expand=True)
        cand_table.add_column("#", style="dim", width=3)
        cand_table.add_column("Score", style="cyan", width=8)
        cand_table.add_column("Candidate Snippet", style="white")
        
        for i, ctx in enumerate(context_details):
            score = ctx.get('rerank_score', ctx.get('score', 0.0))
            text_preview = ctx.get('text', '').replace('\n', ' ').strip()
            if len(text_preview) > 120:
                text_preview = text_preview[:117] + "..."
                
            cand_table.add_row(
                str(i + 1),
                f"{score:.4f}",
                text_preview
            )

        retrieval_panel = Panel(
            cand_table,
            title=f"Top-{len(context_details)} Retrieved Candidates", 
            border_style="blue",
        )
        grid.add_row(retrieval_panel)
        
        # Answer vs Truth
        ans_grid = Table.grid(expand=True, padding=(0, 2))
        ans_grid.add_column(ratio=1)
        ans_grid.add_column(ratio=1)
        
        ans_panel = Panel(Text(answer, style="green"), title="Generated Answer", border_style="green")
        gt_panel = Panel(Text(ground_truth, style="magenta"), title="Ground Truth", border_style="magenta")
        
        ans_grid.add_row(ans_panel, gt_panel)
        
        grid.add_row(ans_grid)
        
        self.console.print(Panel(grid, title=f"[bold]{method_name} - Sample Interaction[/bold]", border_style="white", expand=True))

    def log_method_results(self, method_name: str, df_results: pd.DataFrame):
        """Displays the evaluation metrics for a specific method."""
        # Calculate averages
        avgs = df_results[["faithfulness", "answer_relevancy", "context_precision", "context_recall"]].mean()
        
        table = Table(title=f"Results: {method_name}", box=box.ROUNDED)
        table.add_column("Metric", style="cyan")
        table.add_column("Score", justify="right")
        
        def color_score(score):
            if score >= 0.8: return f"[bold green]{score:.4f}[/bold green]"
            if score >= 0.5: return f"[bold yellow]{score:.4f}[/bold yellow]"
            return f"[bold red]{score:.4f}[/bold red]"

        table.add_row("Faithfulness", color_score(avgs['faithfulness']))
        table.add_row("Answer Relevancy", color_score(avgs['answer_relevancy']))
        table.add_row("Context Precision", color_score(avgs['context_precision']))
        table.add_row("Context Recall", color_score(avgs['context_recall']))
        
        self.console.print(table)

    def save_html_report(self, all_results_df: pd.DataFrame, output_path: Path):
        """Generates a comprehensive HTML report for debugging."""
        
        html_content = f"""
        <html>
        <head>
            <title>LocalRAG Evaluation Report</title>
            <style>
                body {{ font-family: sans-serif; margin: 20px; background: #f0f2f5; }}
                h1, h2 {{ color: #333; }}
                .method-card {{ background: white; padding: 20px; margin-bottom: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                table {{ border-collapse: collapse; width: 100%; margin-top: 10px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; vertical-align: top; }}
                th {{ background-color: #f8f9fa; }}
                .score-high {{ color: green; font-weight: bold; }}
                .score-med {{ color: orange; font-weight: bold; }}
                .score-low {{ color: red; font-weight: bold; }}
                pre {{ white-space: pre-wrap; background: #f4f4f4; padding: 10px; border-radius: 4px; }}
            </style>
        </head>
        <body>
            <h1>Evaluation Report - {datetime.now().strftime('%Y-%m-%d %H:%M')}</h1>
        """
        
        # Summary Table
        html_content += "<h2>Summary</h2><table><thead><tr><th>Method</th><th>Faithfulness</th><th>Relevancy</th><th>Precision</th><th>Recall</th></tr></thead><tbody>"
        
        summary = all_results_df.groupby('method')[["faithfulness", "answer_relevancy", "context_precision", "context_recall"]].mean()
        
        for method, row in summary.iterrows():
            html_content += f"<tr><td>{method}</td><td>{row['faithfulness']:.4f}</td><td>{row['answer_relevancy']:.4f}</td><td>{row['context_precision']:.4f}</td><td>{row['context_recall']:.4f}</td></tr>"
        html_content += "</tbody></table>"
        
        # Detailed Records
        html_content += "<h2>Detailed Interactions</h2>"
        
        for idx, row in all_results_df.iterrows():
            method = row['method']
            q = row['question']
            a = row['answer']
            gt = row['ground_truth']
            ctx = row['contexts']
            
            # Format context as list
            ctx_html = "<ul>" + "".join([f"<li><pre>{c}</pre></li>" for c in ctx]) + "</ul>"
            
            html_content += f"""
            <div class="method-card">
                <h3>[{method}] {q}</h3>
                <table>
                    <tr><td width="15%"><b>Answer</b></td><td>{a}</td></tr>
                    <tr><td><b>Ground Truth</b></td><td>{gt}</td></tr>
                    <tr><td><b>Scores</b></td><td>
                        Faithfulness: {row['faithfulness']:.4f}<br>
                        Relevancy: {row['answer_relevancy']:.4f}<br>
                        Precision: {row['context_precision']:.4f}<br>
                        Recall: {row['context_recall']:.4f}
                    </td></tr>
                    <tr><td><b>Contexts</b></td><td>{ctx_html}</td></tr>
                </table>
            </div>
            """
            
        html_content += "</body></html>"
        
        output_path.write_text(html_content, encoding="utf-8")
        self.console.print(f"\n[bold green]✓ HTML Report saved to:[/bold green] [underline]{output_path}[/underline]")


class AsyncGenerator:
    def __init__(self, model_name="qwen3:14b", base_url=None):
        self.base_url = base_url or "http://192.168.88.21:91"
        self.url = f"{self.base_url}/api/generate"
        self.model_name = model_name

    async def generate(self, question: str, context: list) -> str:
        context_text = "\n\n".join(context)
        full_prompt = (
            f"Используй следующий контекст, чтобы ответить на вопрос.\n"
            f"Контекст: {context_text}\n"
            f"Вопрос: {question}\n"
            f"Ответ:"
        )
        payload = {
            "model": self.model_name,
            "prompt": full_prompt,
            "stream": False
        }
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(self.url, json=payload, timeout=60) as response:
                    response.raise_for_status()
                    data = await response.json()
                    return data.get("response", "Ошибка: Пустой ответ от сервера")
            except Exception as e:
                logger.error(f"Error generating response: {e}")
                return f"Ошибка при обращении к серверу LLM: {e}"

class AsyncEvaluationPipeline:
    def __init__(self):
        self.generator = AsyncGenerator()
        self.evaluator = RagasEvaluator()
        self.reporter = EvaluationReporter(console)

    async def _async_retrieve(self, query, client, embed_model, collection_name, top_k=5):
        return await asyncio.to_thread(retrieve, query, client, embed_model, collection_name, top_k)

    async def _async_rerank(self, query, candidates, top_n=3):
        return await asyncio.to_thread(rerank, query, candidates, top_n)

    async def process_question(self, question, client, embed_model, collection_name):
        # Retrieve
        candidates = await self._async_retrieve(question, client, embed_model, collection_name, top_k=5)
        # Rerank
        final_context = await self._async_rerank(question, candidates, top_n=3)
        # Extract text
        c_texts = [c["text"] for c in final_context]
        # Generate
        answer = await self.generator.generate(question, c_texts)
        
        return {
            "question": question,
            "answer": answer,
            "contexts": c_texts,
            "context_details": final_context
        }

    async def run_method_evaluation(self, method_name, chunk_func, text, questions, ground_truths, progress, overall_task):
        # Create a task for this method in the progress bar
        method_task = progress.add_task(f"[cyan]{method_name}", total=4)
        
        # 1. Chunking
        progress.update(method_task, description=f"[cyan]{method_name}: Chunking...")
        try:
            chunks = await asyncio.to_thread(chunk_func, text)
            progress.advance(method_task)
            
            # Log stats immediately
            self.reporter.log_chunking_stats(method_name, chunks)
            
        except Exception as e:
            logger.error(f"Chunking error in {method_name}: {e}")
            progress.update(method_task, description=f"[red]{method_name}: Failed at Chunking")
            return None
            
        if not chunks:
            progress.update(method_task, description=f"[red]{method_name}: No chunks")
            return None

        # 2. Embedding & Upload
        progress.update(method_task, description=f"[cyan]{method_name}: Embedding...")
        collection_name = f"docs_{method_name.lower().replace(' ', '_')}"
        
        # Suppress stdout from vectorize_and_upload if possible, or just let it be. 
        # Since we want "Detailed", seeing the upload progress is fine, but it breaks the rich layout sometimes.
        # We will just run it.
        client, embed_model = await asyncio.to_thread(vectorize_and_upload, chunks, "sample.md", collection_name)
        progress.advance(method_task)

        # 3. RAG Pipeline (Parallel)
        progress.update(method_task, description=f"[cyan]{method_name}: RAG Pipeline...")
        tasks = [
            self.process_question(q, client, embed_model, collection_name)
            for q in questions
        ]
        results = await asyncio.gather(*tasks)
        
        answers = [r["answer"] for r in results]
        contexts = [r["contexts"] for r in results]
        
        # Log ONE sample interaction for debugging visibility
        if results:
            self.reporter.log_interaction_sample(
                method_name, 
                results[0]['question'], 
                results[0]['context_details'], 
                results[0]['answer'], 
                ground_truths[0]
            )

        progress.advance(method_task)

        # 4. Evaluation
        progress.update(method_task, description=f"[cyan]{method_name}: Evaluating...")
        df_results = await asyncio.to_thread(
            self.evaluator.run_evaluation,
            questions=questions,
            answers=answers,
            contexts=contexts,
            ground_truths=ground_truths,
            max_workers=4 
        )
        
        df_results['method'] = method_name
        progress.advance(method_task)
        progress.update(method_task, visible=False) # Hide completed method task to keep UI clean
        
        # Show mini-summary for this method
        self.reporter.log_method_results(method_name, df_results)
        
        # Advance overall progress
        progress.advance(overall_task)
        
        return df_results

async def main():
    reporter = EvaluationReporter(console)
    reporter.header()

    # Load text
    file_path = PROCESSED_DATA_DIR / "sample.md"
    try:
        markdown_content = await asyncio.to_thread(file_path.read_text, encoding="utf-8")
        console.print(f"[green]✓[/green] Loaded data from [bold]{file_path.name}[/bold] ({len(markdown_content)} chars)")
    except Exception as e:
        logger.error(f"Could not read file {file_path}: {e}")
        return

    # Test Data
    test_questions = [
        "Как называется инструмент визуального моделирования в среде Scilab?",
        "Когда срабатывает блок, имеющий управляющий вход?",
        "Какую функцию выполняет блок ENDBLK?",
    ]
    
    ground_truths = [
        "Инструмент называется Xcos.",
        "Каждый раз при поступлении на него сигнала активации.",
        "Задаёт конечное время моделирования.",
    ]

    pipeline = AsyncEvaluationPipeline()
    
    methods = {
        "Recursive": chunk_recursive,
        "Token": chunk_token,
        "Markdown": chunk_markdown,
        "Sentence Window": chunk_sentence_window,
        "Semantic": chunk_semantic
    }
    
    all_results = []
    
    start_time = time.time()
    
    # Define a clean progress bar layout
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=False
    ) as progress:
        
        overall_task = progress.add_task("[bold]Total Evaluation Progress[/bold]", total=len(methods))
        
        for name, func in methods.items():
            reporter.section(f"Evaluating: {name}")
            res = await pipeline.run_method_evaluation(name, func, markdown_content, test_questions, ground_truths, progress, overall_task)
            if res is not None:
                all_results.append(res)
            
    total_time = time.time() - start_time
    reporter.section("Evaluation Complete", color="green")
    console.print(f"[bold green]Total execution time:[/bold green] {total_time:.2f} seconds")

    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        
        # Save CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("data/evaluation_results")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        csv_path = output_dir / f"eval_results_{timestamp}.csv"
        final_df.to_csv(csv_path, index=False)
        console.print(f"CSV Results saved to [underline]{csv_path}[/underline]")
        
        # Save HTML Report (New Feature)
        html_path = output_dir / f"eval_report_{timestamp}.html"
        reporter.save_html_report(final_df, html_path)
        
        # Final Summary Table
        console.print("\n[bold]Final Comparison:[/bold]")
        summary_table = Table(title="Method Comparison Summary", box=box.HEAVY_HEAD)
        summary_table.add_column("Method", style="cyan", no_wrap=True)
        summary_table.add_column("Faithfulness", justify="right")
        summary_table.add_column("Relevancy", justify="right")
        summary_table.add_column("Precision", justify="right")
        summary_table.add_column("Recall", justify="right")

        avg_df = final_df.groupby('method')[["faithfulness", "answer_relevancy", "context_precision", "context_recall"]].mean().reset_index()
        
        # Find best method based on average of all metrics
        avg_df['mean_score'] = avg_df[["faithfulness", "answer_relevancy", "context_precision", "context_recall"]].mean(axis=1)
        best_method = avg_df.loc[avg_df['mean_score'].idxmax()]['method']
        
        for index, row in avg_df.iterrows():
            style = "bold green" if row['method'] == best_method else None
            summary_table.add_row(
                row['method'],
                f"{row['faithfulness']:.4f}",
                f"{row['answer_relevancy']:.4f}",
                f"{row['context_precision']:.4f}",
                f"{row['context_recall']:.4f}",
                style=style
            )
        console.print(summary_table)
        console.print(f"[bold]Recommended Method:[/bold] [green]{best_method}[/green]")

    else:
        console.print("[red]No results to save.[/red]")

if __name__ == "__main__":
    asyncio.run(main())
