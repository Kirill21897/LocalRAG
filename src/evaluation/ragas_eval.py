import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.llms import LangchainLLMWrapper
from langchain_community.chat_models import ChatOllama
# from langchain_openai import ChatOpenAI
# from langchain_community.embeddings import HuggingFaceEmbeddings # Removed to use shared cache
from src.model_cache import SharedEmbeddings
import config.config as cfg
from ragas.run_config import RunConfig

class RagasEvaluator:
    def __init__(self):
        """
        Initializes the Ragas Evaluator with LLM and Embeddings from config.
        """
        # Setup LLM via ChatOllama to natively support num_ctx and other params
        # This fixes the "unexpected keyword argument 'num_ctx'" error
        
        # Clean base_url (remove /v1 if present as ChatOllama expects base url)
        ollama_url = cfg.OLLAMA_BASE_URL.replace("/v1", "")
        
        self.llm = ChatOllama(
            model=cfg.JUDGE_MODEL,
            base_url=ollama_url,
            temperature=0.0,
            num_ctx=16384,      # Increase context window to 16k tokens
            num_predict=4096,   # Explicit generation limit
            repeat_penalty=1.1, # Prevent infinite loops
            timeout=1200.0      # 20 minutes timeout
        )
        
        # Wrap it for Ragas
        self.judge_llm = LangchainLLMWrapper(self.llm)
        
        # Setup Embeddings using Shared Cache to prevent reloading and OOM
        self.embeddings = SharedEmbeddings()
        
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
