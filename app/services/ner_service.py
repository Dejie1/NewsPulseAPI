"""
Named Entity Recognition service using GLiNER-spaCy.
Detects Magnificent 7 company mentions in article text.
"""

import asyncio
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Optional
from pydantic import BaseModel


# Magnificent 7 company aliases mapping
# Maps various forms of company names/tickers to (normalized_name, ticker)
COMPANY_ALIASES: dict[str, tuple[str, str]] = {
    # Apple
    "apple": ("Apple", "AAPL"),
    "apple inc": ("Apple", "AAPL"),
    "apple inc.": ("Apple", "AAPL"),
    "aapl": ("Apple", "AAPL"),

    # Microsoft
    "microsoft": ("Microsoft", "MSFT"),
    "microsoft corp": ("Microsoft", "MSFT"),
    "microsoft corporation": ("Microsoft", "MSFT"),
    "msft": ("Microsoft", "MSFT"),

    # Google/Alphabet
    "google": ("Google", "GOOGL"),
    "alphabet": ("Google", "GOOGL"),
    "alphabet inc": ("Google", "GOOGL"),
    "alphabet inc.": ("Google", "GOOGL"),
    "googl": ("Google", "GOOGL"),
    "goog": ("Google", "GOOGL"),

    # Amazon
    "amazon": ("Amazon", "AMZN"),
    "amazon.com": ("Amazon", "AMZN"),
    "amazon inc": ("Amazon", "AMZN"),
    "amzn": ("Amazon", "AMZN"),
    "aws": ("Amazon", "AMZN"),
    "amazon web services": ("Amazon", "AMZN"),

    # Meta
    "meta": ("Meta", "META"),
    "meta platforms": ("Meta", "META"),
    "meta platforms inc": ("Meta", "META"),
    "facebook": ("Meta", "META"),

    # Tesla
    "tesla": ("Tesla", "TSLA"),
    "tesla inc": ("Tesla", "TSLA"),
    "tesla motors": ("Tesla", "TSLA"),
    "tsla": ("Tesla", "TSLA"),

    # Nvidia
    "nvidia": ("Nvidia", "NVDA"),
    "nvidia corp": ("Nvidia", "NVDA"),
    "nvidia corporation": ("Nvidia", "NVDA"),
    "nvda": ("Nvidia", "NVDA"),
}

# GLiNER configuration
GLINER_CONFIG = {
    "gliner_model": "urchade/gliner_small-v2",  # Using small model for better compatibility
    "chunk_size": 384,
    "labels": ["company", "organization"],
    "style": "ent",
    "threshold": 0.4,
    "map_location": "cpu",
}


class CompanyEntity(BaseModel):
    """Represents a detected company mention."""
    company_name: str       # Normalized name (e.g., "Google")
    ticker: str             # Stock ticker (e.g., "GOOGL")
    matched_text: str       # Original text that matched
    start_char: int         # Character offset start
    end_char: int           # Character offset end
    confidence: float       # NER confidence score (0-1)
    context_sentence: str   # Sentence containing the mention


class NERService:
    """
    Named Entity Recognition service using GLiNER-spaCy.

    Detects company mentions in text and maps them to the Magnificent 7.
    Uses lazy initialization to avoid loading the model at startup.
    """

    def __init__(self):
        self._nlp = None
        self._initialized = False
        self._initialization_error: Optional[str] = None
        self._executor = ThreadPoolExecutor(max_workers=2)
        self._sentence_splitter = None

    def _ensure_initialized(self):
        """Lazy initialization - loads GLiNER-spaCy model."""
        if self._initialized:
            return

        if self._initialization_error:
            return

        try:
            import spacy
            from gliner_spacy.pipeline import GlinerSpacy

            # Create a blank spaCy pipeline and add GLiNER
            self._nlp = spacy.blank("en")

            # Add GLiNER component
            self._nlp.add_pipe("gliner_spacy", config=GLINER_CONFIG)

            # Add sentencizer for context extraction
            self._nlp.add_pipe("sentencizer")

            self._initialized = True
            print("NER service initialized successfully with GLiNER-spaCy")

        except Exception as e:
            import traceback
            self._initialization_error = str(e)
            print(f"Failed to initialize NER service: {e}")
            traceback.print_exc()

    def _extract_sentence(self, text: str, start: int, end: int) -> str:
        """Extract the sentence containing the entity."""
        # Simple sentence extraction using regex
        # Find sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)

        current_pos = 0
        for sentence in sentences:
            sentence_end = current_pos + len(sentence)
            if current_pos <= start < sentence_end:
                return sentence.strip()
            current_pos = sentence_end + 1  # +1 for the space

        # Fallback: return surrounding context
        context_start = max(0, start - 100)
        context_end = min(len(text), end + 100)
        return text[context_start:context_end].strip()

    def _normalize_company(self, text: str) -> Optional[tuple[str, str]]:
        """
        Normalize detected text to a Magnificent 7 company.

        Returns:
            Tuple of (company_name, ticker) or None if not a tracked company
        """
        normalized = text.lower().strip()

        # Direct match
        if normalized in COMPANY_ALIASES:
            return COMPANY_ALIASES[normalized]

        # Check if any alias is contained in the text
        for alias, (name, ticker) in COMPANY_ALIASES.items():
            if alias in normalized or normalized in alias:
                return (name, ticker)

        return None

    def _extract_sync(self, text: str) -> list[CompanyEntity]:
        """
        Synchronous entity extraction (runs in thread pool).

        Uses GLiNER to detect organizations, then filters to Magnificent 7.
        """
        if not self._initialized:
            return []

        if not text or len(text.strip()) < 10:
            return []

        try:
            # Limit text length for performance
            text = text[:10000]

            # Process with GLiNER
            doc = self._nlp(text)

            entities = []
            seen_positions = set()  # Avoid duplicate entities at same position

            for ent in doc.ents:
                # Check if entity is an organization/company
                if ent.label_ not in ["company", "organization", "COMPANY", "ORGANIZATION", "ORG"]:
                    continue

                # Normalize to Magnificent 7
                normalized = self._normalize_company(ent.text)
                if not normalized:
                    continue

                company_name, ticker = normalized

                # Avoid duplicates at same position
                position_key = (ent.start_char, ent.end_char)
                if position_key in seen_positions:
                    continue
                seen_positions.add(position_key)

                # Extract context sentence
                context = self._extract_sentence(text, ent.start_char, ent.end_char)

                # Calculate confidence (GLiNER provides scores via extension)
                confidence = 0.8  # Default confidence
                if hasattr(ent, '_') and hasattr(ent._, 'score'):
                    confidence = float(ent._.score)

                entities.append(CompanyEntity(
                    company_name=company_name,
                    ticker=ticker,
                    matched_text=ent.text,
                    start_char=ent.start_char,
                    end_char=ent.end_char,
                    confidence=confidence,
                    context_sentence=context[:500]  # Limit context length
                ))

            return entities

        except Exception as e:
            print(f"NER extraction error: {e}")
            return []

    def _extract_with_fallback(self, text: str) -> list[CompanyEntity]:
        """
        Fallback extraction using regex when GLiNER fails or isn't available.
        """
        if not text:
            return []

        entities = []
        seen = set()

        # Build regex pattern from company aliases
        patterns = []
        for alias in COMPANY_ALIASES.keys():
            # Escape special regex characters
            escaped = re.escape(alias)
            patterns.append(escaped)

        # Sort by length (longer patterns first) to match longer names first
        patterns.sort(key=len, reverse=True)
        combined_pattern = r'\b(' + '|'.join(patterns) + r')\b'

        for match in re.finditer(combined_pattern, text, re.IGNORECASE):
            matched_text = match.group(0)
            start = match.start()
            end = match.end()

            # Avoid duplicates
            position_key = (start, end)
            if position_key in seen:
                continue
            seen.add(position_key)

            # Normalize
            normalized = self._normalize_company(matched_text)
            if not normalized:
                continue

            company_name, ticker = normalized
            context = self._extract_sentence(text, start, end)

            entities.append(CompanyEntity(
                company_name=company_name,
                ticker=ticker,
                matched_text=matched_text,
                start_char=start,
                end_char=end,
                confidence=0.9,  # High confidence for exact matches
                context_sentence=context[:500]
            ))

        return entities

    async def extract_companies(self, text: str) -> list[CompanyEntity]:
        """
        Extract Magnificent 7 company mentions from text.

        Args:
            text: The article text to analyze

        Returns:
            List of CompanyEntity objects for detected companies
        """
        self._ensure_initialized()

        if self._initialization_error:
            # Fall back to regex-based extraction
            return self._extract_with_fallback(text)

        try:
            loop = asyncio.get_event_loop()
            entities = await loop.run_in_executor(
                self._executor,
                self._extract_sync,
                text
            )

            # If GLiNER found nothing, try fallback
            if not entities:
                entities = self._extract_with_fallback(text)

            return entities

        except Exception as e:
            print(f"NER extraction failed: {e}")
            return self._extract_with_fallback(text)

    async def extract_companies_batch(
        self,
        texts: list[str]
    ) -> list[list[CompanyEntity]]:
        """
        Extract companies from multiple texts.

        Args:
            texts: List of texts to analyze

        Returns:
            List of entity lists (one per text)
        """
        results = []
        for text in texts:
            entities = await self.extract_companies(text)
            results.append(entities)
        return results

    def get_tracked_companies(self) -> list[dict]:
        """Get list of companies being tracked (Magnificent 7)."""
        companies = {}
        for alias, (name, ticker) in COMPANY_ALIASES.items():
            if ticker not in companies:
                companies[ticker] = {
                    "name": name,
                    "ticker": ticker,
                    "aliases": []
                }
            companies[ticker]["aliases"].append(alias)

        return list(companies.values())


# Global NER service instance
_ner_service: Optional[NERService] = None


def get_ner_service() -> NERService:
    """Get or create the global NER service instance."""
    global _ner_service
    if _ner_service is None:
        _ner_service = NERService()
    return _ner_service
