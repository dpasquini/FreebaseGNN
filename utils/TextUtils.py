import re

class TextUtils:
    """
    Utility class for cleaning and extracting text from LLM outputs.
    This class provides methods to clean the output from models like Gemma, which may include preambles or special tokens, and to extract the relevant response text for further processing.
    """
    
    @staticmethod
    def clean_gemma_output(text: str) -> str:
        """
        Cleans Gemma or similar model outputs by removing preambles like
        'Sure, here is the phrase describing the node "X":' and keeping only
        the functional phrase.
        """
        # Normalize whitespace and quotes
        text = text.strip().strip('"').strip("'").strip()
        # Pattern 1: remove friendly preambles before the actual content
        text = re.sub(
            r'(?is)^(?:sure[,!.\s]*|of course[,!.\s]*|okay[,!.\s]*|here( is|\'s)?( the)?(\s+functional|\s+concise|\s+short)?(\s+phrase|\s+description)?(\s+describing|\s+for)?(\s+the\s+node\s+"[^"]+")?[:\-–—]*\s*)',
            '', text
        )
        # If still contains a preamble mentioning 'node "..."' before the real text
        text = re.sub(
            r'(?is)^the\s+phrase\s+describing\s+the\s+node\s+"[^"]+"\s*[:\-–—]*\s*',
            '',
            text
        )
        # Clean any remaining leading/trailing quotes or spaces
        text = text.strip().strip('"').strip("'").strip()
        # If there are multiple lines, keep the last non-empty one (the actual phrase)
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        if len(lines) > 1:
            text = lines[-1]
        return text

    @staticmethod
    def extract_model_response(raw_output: str) -> str:
        """
        Extracts the model's response from the full generated text for Gemma models.
        """
        # Remove special tokens
        text = re.sub(r"<[^>]+>", "", raw_output)  # removes <bos>, <start_of_turn>, </end_of_turn>, etc.

        # Split into turns
        turns = text.split("model")
        if len(turns) < 2:
            return text.strip()

        # Get the last model response
        response = turns[-1]

        # Trim remaining noise and whitespace
        response = response.strip().strip('"').strip()
        return response

    @staticmethod
    def _normalize_for_dedup(text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text

    @staticmethod
    def _is_low_value_fact(text: str) -> bool:
        low_value_patterns = [
            r"\bimportant\b",
            r"\bnotable\b",
            r"\bwell\s*known\b",
            r"\bfamous\b",
            r"\bentity\b",
            r"\bthing\b",
            r"\bobject\b",
        ]
        normalized = TextUtils._normalize_for_dedup(text)
        if len(normalized.split()) < 3:
            return True
        return any(re.search(p, normalized) for p in low_value_patterns)

    @staticmethod
    def compress_description(text: str, max_facts: int = 3, max_words_per_fact: int = 16) -> str:
        """
        Post-processes LLM descriptions to reduce noise/redundancy:
        - splits into candidate facts
        - removes low-value/generic facts
        - deduplicates near-identical facts
        - enforces compact length
        """
        if not isinstance(text, str):
            return text

        raw = text.strip()
        if not raw:
            return raw

        bullet_like = re.split(r"\n+|\s*[\u2022\-]\s+", raw)
        if len([c for c in bullet_like if c and c.strip()]) <= 1:
            candidates = re.split(r"[\.;\n]+", raw)
        else:
            candidates = bullet_like

        unique = []
        seen = set()
        for c in candidates:
            c = c.strip().strip('"').strip("'")
            if not c:
                continue

            if TextUtils._is_low_value_fact(c):
                continue

            norm = TextUtils._normalize_for_dedup(c)
            if norm in seen:
                continue
            seen.add(norm)

            words = c.split()
            if len(words) > max_words_per_fact:
                c = " ".join(words[:max_words_per_fact]).rstrip(" ,;:")

            unique.append(c)
            if len(unique) >= max_facts:
                break

        if not unique:
            fallback = TextUtils.clean_gemma_output(raw)
            fallback_words = fallback.split()
            if len(fallback_words) > max_words_per_fact:
                fallback = " ".join(fallback_words[:max_words_per_fact])
            return fallback.strip()

        return " ; ".join(unique)