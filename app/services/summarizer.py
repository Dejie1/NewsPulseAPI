"""
Summarization service with multiple algorithm options.
Uses sumy library for proper extractive summarization.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from app.models import Article, SummarizationResponse

logger = logging.getLogger(__name__)


class BaseSummarizer(ABC):
    """
    Abstract base class for summarizers.
    Implement this to add different summarization backends.
    """

    @abstractmethod
    async def summarize(self, text: str, max_length: int = 200) -> str:
        """
        Summarize the given text.

        Args:
            text: The text to summarize
            max_length: Target summary length in words

        Returns:
            Summarized text
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the summarizer for tracking."""
        pass


class NoOpSummarizer(BaseSummarizer):
    """
    Fallback summarizer that returns truncated content.
    Use this when no other summarizer is available.
    """

    async def summarize(self, text: str, max_length: int = 200) -> str:
        """Return first N words of text."""
        words = text.split()
        if len(words) <= max_length:
            return text
        return " ".join(words[:max_length]) + "..."

    @property
    def name(self) -> str:
        return "truncate"


class SumySummarizer(BaseSummarizer):
    """
    Extractive summarizer using sumy library.
    Supports multiple algorithms: LSA, TextRank, LexRank, Luhn.

    This is a proper NLP-based summarizer that:
    1. Tokenizes text into sentences
    2. Builds a representation (TF-IDF, graph, etc.)
    3. Ranks sentences by importance
    4. Returns top-ranked sentences
    """

    ALGORITHMS = ["lsa", "textrank", "lexrank", "luhn"]

    def __init__(self, algorithm: str = "lsa"):
        """
        Initialize with a specific algorithm.

        Args:
            algorithm: One of "lsa", "textrank", "lexrank", "luhn"
                - lsa: Latent Semantic Analysis (good general purpose)
                - textrank: Graph-based, similar to Google PageRank
                - lexrank: Graph-based, uses cosine similarity
                - luhn: Classic, based on word frequency
        """
        if algorithm not in self.ALGORITHMS:
            raise ValueError(f"Algorithm must be one of: {self.ALGORITHMS}")

        self.algorithm = algorithm
        self._executor = ThreadPoolExecutor(max_workers=2)
        self._nltk_downloaded = False

    def shutdown(self) -> None:
        """Shut down the thread pool executor."""
        self._executor.shutdown(wait=False)

    def _ensure_nltk_data(self):
        """Download required NLTK data if not present."""
        if self._nltk_downloaded:
            return

        import nltk
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)

        try:
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            nltk.download('punkt_tab', quiet=True)

        self._nltk_downloaded = True

    def _get_summarizer(self):
        """Get the appropriate sumy summarizer based on algorithm."""
        if self.algorithm == "lsa":
            from sumy.summarizers.lsa import LsaSummarizer
            return LsaSummarizer()
        elif self.algorithm == "textrank":
            from sumy.summarizers.text_rank import TextRankSummarizer
            return TextRankSummarizer()
        elif self.algorithm == "lexrank":
            from sumy.summarizers.lex_rank import LexRankSummarizer
            return LexRankSummarizer()
        elif self.algorithm == "luhn":
            from sumy.summarizers.luhn import LuhnSummarizer
            return LuhnSummarizer()

    def _summarize_sync(self, text: str, sentence_count: int) -> str:
        """Synchronous summarization (runs in thread pool)."""
        self._ensure_nltk_data()

        from sumy.parsers.plaintext import PlaintextParser
        from sumy.nlp.tokenizers import Tokenizer
        from sumy.nlp.stemmers import Stemmer
        from sumy.utils import get_stop_words

        LANGUAGE = "english"

        try:
            # Parse the text
            parser = PlaintextParser.from_string(text, Tokenizer(LANGUAGE))

            # Get the summarizer
            summarizer = self._get_summarizer()
            summarizer.stop_words = get_stop_words(LANGUAGE)

            # Generate summary
            summary_sentences = summarizer(parser.document, sentence_count)

            # Join sentences
            summary = " ".join(str(sentence) for sentence in summary_sentences)

            return summary if summary else text[:500]

        except Exception as e:
            # Fallback to simple truncation if sumy fails
            logger.error("Sumy summarization failed: %s", e)
            words = text.split()
            return " ".join(words[:100]) + "..."

    async def summarize(self, text: str, max_length: int = 200) -> str:
        """
        Summarize text using the configured algorithm.

        Args:
            text: Text to summarize
            max_length: Target length in words (converted to ~sentence count)
        """
        if not text or len(text.strip()) < 100:
            return text

        # Estimate sentence count from word count
        # Average sentence is ~15-20 words
        sentence_count = max(2, max_length // 20)

        # Run in thread pool since sumy is blocking
        loop = asyncio.get_event_loop()
        summary = await loop.run_in_executor(
            self._executor,
            self._summarize_sync,
            text,
            sentence_count
        )

        return summary

    @property
    def name(self) -> str:
        return f"sumy-{self.algorithm}"


# =============================================================================
# AI SUMMARIZER TEMPLATES
# Uncomment and configure one of these when you're ready to add AI
# =============================================================================

# class OpenAISummarizer(BaseSummarizer):
#     """
#     OpenAI GPT-based summarizer.
#     Requires: pip install openai
#     """
#
#     def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
#         from openai import AsyncOpenAI
#         self.client = AsyncOpenAI(api_key=api_key)
#         self.model = model
#
#     async def summarize(self, text: str, max_length: int = 200) -> str:
#         response = await self.client.chat.completions.create(
#             model=self.model,
#             messages=[
#                 {
#                     "role": "system",
#                     "content": f"Summarize the following article in about {max_length} words. "
#                                "Be concise and capture the key points."
#                 },
#                 {"role": "user", "content": text}
#             ],
#             max_tokens=max_length * 2,
#             temperature=0.3
#         )
#         return response.choices[0].message.content
#
#     @property
#     def name(self) -> str:
#         return "openai"


# class ClaudeSummarizer(BaseSummarizer):
#     """
#     Anthropic Claude-based summarizer.
#     Requires: pip install anthropic
#     """
#
#     def __init__(self, api_key: str, model: str = "claude-3-haiku-20240307"):
#         from anthropic import AsyncAnthropic
#         self.client = AsyncAnthropic(api_key=api_key)
#         self.model = model
#
#     async def summarize(self, text: str, max_length: int = 200) -> str:
#         response = await self.client.messages.create(
#             model=self.model,
#             max_tokens=max_length * 2,
#             messages=[
#                 {
#                     "role": "user",
#                     "content": f"Summarize this article in about {max_length} words:\n\n{text}"
#                 }
#             ]
#         )
#         return response.content[0].text
#
#     @property
#     def name(self) -> str:
#         return "claude"


# class OllamaSummarizer(BaseSummarizer):
#     """
#     Local Ollama-based summarizer (free, runs locally).
#     Requires: Ollama running locally with a model pulled
#     """
#
#     def __init__(self, model: str = "llama2", base_url: str = "http://localhost:11434"):
#         import httpx
#         self.model = model
#         self.base_url = base_url
#         self.client = httpx.AsyncClient()
#
#     async def summarize(self, text: str, max_length: int = 200) -> str:
#         response = await self.client.post(
#             f"{self.base_url}/api/generate",
#             json={
#                 "model": self.model,
#                 "prompt": f"Summarize this article in about {max_length} words:\n\n{text}",
#                 "stream": False
#             },
#             timeout=60.0
#         )
#         return response.json()["response"]
#
#     @property
#     def name(self) -> str:
#         return "ollama"


class SummarizationService:
    """
    Main summarization service.
    Coordinates between content extraction and summarization.
    """

    def __init__(self, summarizer: Optional[BaseSummarizer] = None):
        # Default to sumy LSA summarizer
        self.summarizer = summarizer or SumySummarizer(algorithm="lsa")

    async def summarize_article(
        self,
        article: Article,
        max_length: int = 200
    ) -> SummarizationResponse:
        """
        Summarize an article's content.

        Args:
            article: Article to summarize (must have content extracted)
            max_length: Target summary length in words

        Returns:
            SummarizationResponse with summary and metadata
        """
        # Use content if available, otherwise fall back to description
        text_to_summarize = article.content or article.description or ""

        if not text_to_summarize:
            return SummarizationResponse(
                original_title=article.title,
                summary="No content available to summarize.",
                source=article.source,
                summarized_by=self.summarizer.name,
                original_length=0,
                summary_length=0
            )

        # Generate summary
        summary = await self.summarizer.summarize(text_to_summarize, max_length)

        return SummarizationResponse(
            original_title=article.title,
            summary=summary,
            source=article.source,
            summarized_by=self.summarizer.name,
            original_length=len(text_to_summarize.split()),
            summary_length=len(summary.split())
        )

    def set_summarizer(self, summarizer: BaseSummarizer) -> None:
        """
        Switch to a different summarizer.
        Use this to change algorithm or enable AI summarization.
        """
        self.summarizer = summarizer


# Global summarization service instance
_summarization_service: Optional[SummarizationService] = None


def get_summarization_service() -> SummarizationService:
    """Get or create the global summarization service."""
    global _summarization_service
    if _summarization_service is None:
        _summarization_service = SummarizationService()
    return _summarization_service


def configure_summarizer(algorithm: str = "lsa") -> None:
    """
    Configure the summarizer algorithm.

    Args:
        algorithm: One of "lsa", "textrank", "lexrank", "luhn"
            - lsa: Latent Semantic Analysis (default, good general purpose)
            - textrank: Graph-based ranking (like Google PageRank)
            - lexrank: Graph-based with cosine similarity
            - luhn: Classic frequency-based

    Example:
        configure_summarizer("textrank")
    """
    service = get_summarization_service()
    service.set_summarizer(SumySummarizer(algorithm=algorithm))


def configure_ai_summarizer(provider: str, api_key: str, **kwargs) -> None:
    """
    Configure an AI summarizer.

    Args:
        provider: "openai", "claude", or "ollama"
        api_key: API key (not needed for ollama)
        **kwargs: Additional provider-specific options

    Example:
        configure_ai_summarizer("openai", "sk-...")
        configure_ai_summarizer("claude", "sk-ant-...")
        configure_ai_summarizer("ollama", "", model="llama2")
    """
    service = get_summarization_service()

    # Uncomment the provider you want to use:

    # if provider == "openai":
    #     service.set_summarizer(OpenAISummarizer(api_key, **kwargs))
    # elif provider == "claude":
    #     service.set_summarizer(ClaudeSummarizer(api_key, **kwargs))
    # elif provider == "ollama":
    #     service.set_summarizer(OllamaSummarizer(**kwargs))

    # For now, fall back to sumy
    logger.info("AI summarizer '%s' not configured. Using sumy-lsa instead.", provider)
    service.set_summarizer(SumySummarizer(algorithm="lsa"))
