import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.llms import llm_factory
from langchain_community.embeddings import HuggingFaceEmbeddings
from openai import AsyncOpenAI
import config.config as cfg
from ragas.run_config import RunConfig

class RagasEvaluator:
    def __init__(self):
        """
        Initializes the Ragas Evaluator with LLM and Embeddings from config.
        """
        # Setup LLM client with EXTREMELY high timeout
        self.client = AsyncOpenAI(
            base_url=cfg.OLLAMA_BASE_URL,
            api_key=cfg.OLLAMA_API_KEY,
            timeout=1200.0 # 20 minutes timeout
        )
        
        # Create LLM wrapper
        self.judge_llm = llm_factory(cfg.JUDGE_MODEL, client=self.client)
        
        # Setup Embeddings
        self.embeddings = HuggingFaceEmbeddings(model_name=cfg.EMBEDDING_MODEL)
        
        # Define metrics list
        self.metrics = [
            faithfulness, 
            answer_relevancy, 
            context_precision, 
            context_recall
        ]
        
        # Bind models to metrics
        self._configure_metrics()

    def _configure_metrics(self):
        """
        Assigns the configured LLM and Embeddings to each metric.
        """
        for metric in self.metrics:
            if hasattr(metric, 'llm'):
                metric.llm = self.judge_llm
            if hasattr(metric, 'embeddings'):
                metric.embeddings = self.embeddings

    def run_evaluation(self, questions, answers, contexts, ground_truths):
        """
        Runs Ragas evaluation on the provided data.
        
        Args:
            questions (list): List of user questions.
            answers (list): List of generated answers.
            contexts (list): List of lists of retrieved context strings.
            ground_truths (list): List of ground truth answers.
            
        Returns:
            pd.DataFrame: DataFrame containing the evaluation results with numeric metrics.
        """
        # Create dataset
        data = {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths
        }
        dataset = Dataset.from_dict(data)
        
        print("Starting Ragas evaluation pipeline...")
        # Configure run to use single thread to avoid local LLM overload
        # and set a high timeout for the run configuration itself if applicable
        my_run_config = RunConfig(timeout=1200, max_workers=1)
        
        results = evaluate(
            dataset=dataset,
            metrics=self.metrics,
            run_config=my_run_config,
            raise_exceptions=False, 
        )
        
        # Convert to pandas
        df = results.to_pandas()
        
        # Ensure all metric columns are numeric
        metric_names = [m.name for m in self.metrics]
        for col in metric_names:
            if col in df.columns:
                # Convert to numeric, coercing errors to NaN
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
        return df

    def save_results(self, df, output_path):
        """
        Saves the evaluation results to a CSV file.
        """
        df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")
