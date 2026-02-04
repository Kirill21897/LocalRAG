import asyncio
import aiohttp
import time
import pandas as pd
from typing import List, Dict, Any, Callable
from pathlib import Path
import logging
import nest_asyncio

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

# Rich imports
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
from rich.table import Table
from rich.panel import Panel
from rich.logging import RichHandler
from rich import print as rprint

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
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger("rich")
console = Console()

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
            "contexts": c_texts
        }

    async def run_method_evaluation(self, method_name, chunk_func, text, questions, ground_truths, progress, overall_task):
        console.print(Panel(f"[bold blue]Testing Method: {method_name}[/bold blue]", expand=False))
        
        # Create a task for this method
        method_task = progress.add_task(f"[cyan]{method_name}: Processing...", total=4)

        # 1. Chunking
        progress.update(method_task, description=f"[cyan]{method_name}: Chunking...")
        try:
            chunks = await asyncio.to_thread(chunk_func, text)
            logger.info(f"Method {method_name}: Generated {len(chunks)} chunks.")
            progress.advance(method_task)
        except Exception as e:
            logger.error(f"Chunking error in {method_name}: {e}")
            progress.update(method_task, description=f"[red]{method_name}: Failed at Chunking")
            return None
            
        if not chunks:
            logger.error(f"No chunks generated for {method_name}.")
            progress.update(method_task, description=f"[red]{method_name}: No chunks")
            return None

        # 2. Embedding & Upload
        progress.update(method_task, description=f"[cyan]{method_name}: Embedding...")
        collection_name = f"docs_{method_name.lower().replace(' ', '_')}"
        # vectorize_and_upload returns client, embed_model
        # Note: vectorize_and_upload prints to stdout, which might interfere with rich progress bar.
        # Ideally we should silence it or redirect it, but for now we let it be.
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
        progress.advance(method_task)

        # 4. Evaluation
        progress.update(method_task, description=f"[cyan]{method_name}: Evaluating...")
        # Run evaluation in thread because it might block internally
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
        progress.update(method_task, description=f"[green]{method_name}: Done", completed=4)
        
        # Advance overall progress
        progress.advance(overall_task)
        
        return df_results

async def main():
    console.clear()
    console.print(Panel.fit("[bold green]LocalRAG Evaluation Pipeline[/bold green]", subtitle="v2.0 Async & Rich"))

    # Load text
    file_path = PROCESSED_DATA_DIR / "sample.md"
    try:
        markdown_content = await asyncio.to_thread(file_path.read_text, encoding="utf-8")
        console.print(f"[green]✓[/green] Loaded data from [bold]{file_path.name}[/bold]")
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
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=False
    ) as progress:
        
        overall_task = progress.add_task("[bold]Total Progress[/bold]", total=len(methods))
        
        for name, func in methods.items():
            res = await pipeline.run_method_evaluation(name, func, markdown_content, test_questions, ground_truths, progress, overall_task)
            if res is not None:
                all_results.append(res)
            
    total_time = time.time() - start_time
    console.print(f"\n[bold green]Total execution time:[/bold green] {total_time:.2f} seconds")

    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        output_path = Path("data/evaluation_results/full_comparison_results_async.csv")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        final_df.to_csv(output_path, index=False)
        console.print(f"Results saved to [underline]{output_path}[/underline]")
        
        # Display Summary Table
        table = Table(title="Evaluation Summary")
        table.add_column("Method", style="cyan", no_wrap=True)
        table.add_column("Faithfulness", justify="right")
        table.add_column("Ans Relevancy", justify="right")
        table.add_column("Ctx Precision", justify="right")
        table.add_column("Ctx Recall", justify="right")

        for index, row in final_df.iterrows():
            table.add_row(
                row['method'],
                f"{row['faithfulness']:.4f}",
                f"{row['answer_relevancy']:.4f}",
                f"{row['context_precision']:.4f}",
                f"{row['context_recall']:.4f}"
            )
            
        console.print(table)
        
        # Calculate averages per method
        avg_df = final_df.groupby('method')[["faithfulness", "answer_relevancy", "context_precision", "context_recall"]].mean().reset_index()
        
        avg_table = Table(title="Average Scores per Method")
        avg_table.add_column("Method", style="magenta", no_wrap=True)
        avg_table.add_column("Avg Faithfulness", justify="right")
        avg_table.add_column("Avg Ans Relevancy", justify="right")
        avg_table.add_column("Avg Ctx Precision", justify="right")
        avg_table.add_column("Avg Ctx Recall", justify="right")
        
        for index, row in avg_df.iterrows():
            avg_table.add_row(
                row['method'],
                f"{row['faithfulness']:.4f}",
                f"{row['answer_relevancy']:.4f}",
                f"{row['context_precision']:.4f}",
                f"{row['context_recall']:.4f}"
            )
        console.print(avg_table)

    else:
        console.print("[red]No results to save.[/red]")

if __name__ == "__main__":
    asyncio.run(main())
