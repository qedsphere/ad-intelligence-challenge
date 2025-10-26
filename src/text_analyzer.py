"""
Lightweight text analysis for transcripts: keywords, entities, CTA, intent.
Fast and accurate: batch TF-IDF, domain stopwords, CTA exclusion, phrases.
"""

from __future__ import annotations

import re
import math
from collections import Counter, defaultdict
from typing import Dict, List
from typing import Optional

# Generic stopwords (algorithm will handle domain-specific filtering)
STOP_WORDS = {
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he',
    'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 'to', 'was', 'will', 'with',
    'can', 'we', 'you', 'your', 'our', 'they', 'them', 'this', 'these', 'those', 'all',
    'what', 'when', 'how', 'why', 'who', 'which', 'where', 'here', 'there', 'when',
    'because', 'although', 'though', 'even', 'if', 'unless', 'whether', 'once', 'twice'
}
DOMAIN_STOP_WORDS = {
    'now', 'today', 'new', 'best', 'deal', 'offer', 'save', 'off', 'percent', 'limited',
    'free', 'click', 'shop', 'get', 'learn', 'more', 'only', 'great', 'just'
}

CTA_PATTERNS = [
    r'\b(buy|purchase|order|shop|get|download|subscribe|sign up|join|try|call|visit|click|register|book|reserve)\b',
    r'\b(now|today|free|limited|offer|deal|sale|discount)\b',
    r'\b(learn more|find out|discover|explore)\b',
]

# Optional semantic keyword backends (local)
try:
    from keybert import KeyBERT  # type: ignore
    from sentence_transformers import SentenceTransformer  # type: ignore
    _KB_MODEL: Optional[KeyBERT] = None
except Exception:
    KeyBERT = None  # type: ignore
    SentenceTransformer = None  # type: ignore
    _KB_MODEL = None

try:
    import yake  # type: ignore
except Exception:
    yake = None  # type: ignore


def _tokenize(text: str) -> List[str]:
    return re.findall(r"\b[a-z]{3,}\b", text.lower())


def _simple_stem(token: str) -> str:
    if len(token) <= 3:
        return token
    for suf in ("ing", "ers", "ies", "ed", "es", "s"):
        if token.endswith(suf) and len(token) - len(suf) >= 3:
            return token[:-3] + "y" if suf == "ies" else token[: -len(suf)]
    return token


def _extract_cta(text: str) -> Dict[str, object]:
    if not text or not text.strip():
        return {'detected': False, 'phrases': []}
    tl = text.lower()
    phrases = []
    for pat in CTA_PATTERNS:
        phrases.extend(re.findall(pat, tl))
    flat = []
    for m in phrases:
        if isinstance(m, tuple):
            flat.extend([w for w in m if w])
        else:
            flat.append(m)
    uniq = list(dict.fromkeys(flat))[:3]
    return {'detected': len(uniq) > 0, 'phrases': uniq}


def _cta_words(text: str) -> set[str]:
    tl = text.lower()
    found = []
    for pat in CTA_PATTERNS:
        found.extend(re.findall(pat, tl))
    words = set()
    for m in found:
        if isinstance(m, tuple):
            m = " ".join(w for w in m if w)
        words.update(re.findall(r"\b[a-z]{3,}\b", str(m)))
    return words


def _extract_entities(text: str) -> List[str]:
    # Extract proper nouns: consecutive capitalized spans (1–3 words), algorithmic filtering
    spans = re.findall(r"\b([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,2})\b", text)
    cleaned = []
    for s in spans:
        clean = re.sub(r"[^A-Za-z\s]", "", s).strip()
        if not clean or len(clean.split()) < 1 or len(clean) < 4:
            continue
        # Skip sentence fragments (words that typically start sentences)
        words = clean.split()
        if any(w.lower() in {'it', 'and', 'but', 'or', 'for', 'the', 'this', 'that', 'with', 'from'} for w in words):
            continue
        # Skip if too generic (common patterns)
        clean_lower = clean.lower()
        if clean_lower in {'bring it', 'look at', 'together let', 'make it', 'take it'}:
            continue
        cleaned.append(clean)
    return list(dict.fromkeys(cleaned))


def _build_df_map(texts: List[str]) -> Dict[str, int]:
    df: Dict[str, int] = defaultdict(int)
    for t in texts:
        toks = {_simple_stem(tok) for tok in _tokenize(t)}
        for tok in toks:
            df[tok] += 1
    return df


def _idf(term: str, df_map: Dict[str, int] | None, num_docs: int | None) -> float:
    if not df_map or not num_docs:
        return 1.0
    return 1.0 + math.log((num_docs + 1) / (max(1, df_map.get(term, 1)) + 1))


def extract_keywords(text: str, top_n: int = 5, df_map: Dict[str, int] | None = None, num_docs: int | None = None) -> List[str]:
    if not text or not text.strip():
        return []

    tokens = _tokenize(text)
    if not tokens:
        return []

    # Build stop set and stemmed unigrams
    stop = STOP_WORDS | DOMAIN_STOP_WORDS | _cta_words(text)
    filtered = [_simple_stem(t) for t in tokens if t not in stop]
    if not filtered:
        return []

    # Unigram TF-IDF with entity boost and slight position bonus
    tf = Counter(filtered)
    entities = {e.lower() for e in _extract_entities(text)}
    first_occ: Dict[str, int] = {}
    for i, tok in enumerate(filtered):
        first_occ.setdefault(tok, i)
    doc_len = max(1, len(filtered))

    uni_scores: Dict[str, float] = {}
    for term, freq in tf.items():
        score = freq * _idf(term, df_map, num_docs)
        # Algorithmic length preference: longer words often more specific
        score *= 1.0 + math.log(len(term)) / 4.0
        # Stronger entity boost
        score *= 1.0 + (0.5 if term in entities else 0.0)
        # Position bonus: earlier occurrence gets slight boost
        pos_bonus = 1.0 + 0.15 * (1.0 - (first_occ.get(term, 0) / doc_len))
        # Specificity bonus: prefer words that aren't too rare or too common
        term_df_ratio = df_ratio(term)
        specificity_bonus = 1.0
        if 0.05 < term_df_ratio < 0.4:  # goldilocks zone: not too rare, not too common
            specificity_bonus = 1.2
        uni_scores[term] = score * pos_bonus * specificity_bonus

    # Phrase candidates: bigrams + trigrams, algorithmic filtering
    bigrams = [f"{a} {b}" for a, b in zip(tokens, tokens[1:]) if a not in stop and b not in stop]
    trigrams = [f"{a} {b} {c}" for a, b, c in zip(tokens, tokens[1:], tokens[2:]) if a not in stop and b not in stop and c not in stop]
    # Filter phrases that are too generic (algorithmic approach)
    bi_tf = Counter(bigrams)
    tri_tf = Counter(trigrams)
    # Only keep phrases that appear in this document and aren't too generic
    for ph, freq in list(bi_tf.items()):
        if freq < 1 or df_ratio(ph.split()[0]) > 0.6:  # first word too common
            del bi_tf[ph]
    for ph, freq in list(tri_tf.items()):
        if freq < 1 or any(df_ratio(p) > 0.7 for p in ph.split()):  # any word too common
            del tri_tf[ph]

    def phrase_score(parts: List[str], freq: int) -> float:
        base = sum(_idf(_simple_stem(p), df_map, num_docs) for p in parts) / len(parts)
        boost = 2.0 if len(parts) == 2 else 2.2  # Higher preference for phrases
        if any(p.lower() in entities for p in parts):
            boost += 0.3  # Stronger entity boost
        # Penalize phrases made entirely of domain/CTA words
        if all(p.lower() in (DOMAIN_STOP_WORDS | _cta_words(text)) for p in parts):
            boost *= 0.5
        # Algorithmic phrase quality: penalize phrases with too many generic words
        generic_count = sum(1 for p in parts if df_ratio(p) > 0.5)  # words that appear in >50% of docs
        if generic_count / len(parts) > 0.6:  # >60% of phrase is generic words
            boost *= 0.4
        # Prefer phrases with mixed specificity (not all generic, not all rare)
        rare_count = sum(1 for p in parts if df_ratio(p) < 0.1)  # words that appear in <10% of docs
        if rare_count == 0:  # no specific words at all
            boost *= 0.5
        return freq * base * boost

    bi_scores = {ph: phrase_score(ph.split(" "), f) for ph, f in bi_tf.items()}
    tri_scores = {ph: phrase_score(ph.split(" "), f) for ph, f in tri_tf.items()}

    # Algorithmic filtering: drop corpus-generic terms (>20% of docs)
    def df_ratio(tok: str) -> float:
        if not df_map or not num_docs:
            return 0.0
        return df_map.get(_simple_stem(tok), 0) / max(1, num_docs)

    # Algorithmic entity detection: find capitalized spans that appear in multiple docs
    def is_potential_brand(span: str) -> bool:
        if not df_map or not num_docs:
            return False
        # Check each word in the span individually
        words = span.lower().split()
        word_dfs = [df_map.get(w, 0) for w in words]
        # Span is a potential brand if words appear in multiple docs but not ubiquitous
        avg_df = sum(word_dfs) / len(word_dfs)
        return avg_df >= 2 and avg_df / num_docs <= 0.6

    cands: List[tuple[str, float, bool]] = []
    for t, s in uni_scores.items():
        # Stricter corpus filtering + length preference
        if df_ratio(t) <= 0.2 and len(t) >= 4:  # only longer, less common words
            cands.append((t, s, False))
    for ph, s in {**bi_scores, **tri_scores}.items():
        parts = ph.split(" ")
        # Phrases must have at least one specific word and not be too generic
        if (max(df_ratio(p) for p in parts) <= 0.7 and
            len(ph) >= 6 and  # prefer longer phrases
            any(len(p) >= 5 for p in parts)):  # at least one specific word
            cands.append((ph, s, True))
    if not cands:
        return []

    cands.sort(key=lambda x: x[1], reverse=True)
    top = cands[0][1]
    threshold = 0.5 * top
    selected: List[str] = []
    masked: set[str] = set()
    for text_item, score, is_phrase in cands:
        if score < threshold:
            continue
        if is_phrase:
            parts = [ _simple_stem(p) for p in text_item.split(" ") ]
            if any(p in masked for p in parts):
                continue
            selected.append(text_item)
            masked.update(parts)
        else:
            if _simple_stem(text_item) in masked:
                continue
            selected.append(text_item)
        if len(selected) >= top_n:
            break

    # Restore best casing from original text
    def best_case(term: str) -> str:
        if " " in term:
            return " ".join(best_case(p) for p in term.split(" "))
        m = re.search(rf"\b{re.escape(term)}\b", text, flags=re.IGNORECASE)
        return m.group(0) if m else term

    return [best_case(s) for s in selected]


# ---------- Semantic keywording (KeyBERT / YAKE) ----------

def _get_keybert() -> Optional[KeyBERT]:
    global _KB_MODEL
    if KeyBERT is None or SentenceTransformer is None:
        return None
    if _KB_MODEL is None:
        # Small, fast local model
        st_model = SentenceTransformer('all-MiniLM-L6-v2')
        _KB_MODEL = KeyBERT(model=st_model)
    return _KB_MODEL


def extract_keywords_keybert(text: str, top_n: int = 5) -> List[str]:
    kb = _get_keybert()
    if kb is None or not text or not text.strip():
        return []
    try:
        # Get more than needed, we'll post-filter
        cand = kb.extract_keywords(
            text,
            keyphrase_ngram_range=(1, 3),
            stop_words='english',
            use_mmr=True,
            diversity=0.7,
            top_n=max(15, top_n * 3),
        )
        # cand is List[(phrase, score)] with higher score = better
        phrases = [p for p, _ in cand if p]
        # Prefer multi-word phrases, keep unique order
        uniq: List[str] = []
        seen = set()
        # First add 2-3 word phrases
        for p in phrases:
            if ((' ' in p) and (p not in seen)):
                uniq.append(p)
                seen.add(p)
                if len(uniq) >= top_n:
                    break
        # Then fill with strong unigrams
        if len(uniq) < top_n:
            for p in phrases:
                if (' ' not in p) and (p not in seen):
                    uniq.append(p)
                    seen.add(p)
                    if len(uniq) >= top_n:
                        break
        return uniq[:top_n]
    except Exception:
        return []


def extract_keywords_yake(text: str, top_n: int = 5) -> List[str]:
    if yake is None or not text or not text.strip():
        return []
    try:
        kw = yake.KeywordExtractor(lan='en', n=3, top=top_n * 3, dedupLim=0.9)
        cands = kw.extract_keywords(text)
        # cands: List[(phrase, score)] lower score = better in YAKE
        cands.sort(key=lambda x: x[1])
        phrases = [p for p, _ in cands]
        uniq: List[str] = []
        seen = set()
        # Prefer phrases (2-3 words), then unigrams
        for p in phrases:
            if ((' ' in p) and (p not in seen)):
                uniq.append(p)
                seen.add(p)
                if len(uniq) >= top_n:
                    break
        if len(uniq) < top_n:
            for p in phrases:
                if (' ' not in p) and (p not in seen):
                    uniq.append(p)
                    seen.add(p)
                    if len(uniq) >= top_n:
                        break
        return uniq[:top_n]
    except Exception:
        return []


def classify_intent(text: str) -> Dict[str, object]:
    if not text or not text.strip():
        return {'primary': 'unknown', 'confidence': 0.0}
    tl = text.lower()
    groups = {
        'persuasive': ['buy', 'purchase', 'order', 'get', 'try', 'deal', 'offer', 'sale', 'limited', 'free'],
        'informative': ['learn', 'discover', 'understand', 'know', 'information', 'feature', 'benefit'],
        'emotional': ['love', 'enjoy', 'feel', 'amazing', 'wonderful', 'perfect', 'beautiful', 'dream'],
        'problem_solving': ['problem', 'solution', 'fix', 'help', 'relief', 'issue', 'trouble'],
    }
    scores = {k: sum(1 for w in vs if w in tl) for k, vs in groups.items()}
    scores = {k: v for k, v in scores.items() if v > 0}
    if not scores:
        return {'primary': 'informative', 'confidence': 0.3}
    primary = max(scores, key=scores.get)
    total = sum(scores.values())
    return {'primary': primary, 'confidence': round(scores[primary]/max(1,total), 2)}


def analyze_text(text: str, df_map: Dict[str, int] | None = None, num_docs: int | None = None) -> Dict[str, object]:
    if not text or not text.strip():
        return {'keywords': [], 'entities': [], 'cta': {'detected': False, 'phrases': []}, 'intent': {'primary': 'unknown', 'confidence': 0.0}}
    ents = _extract_entities(text)
    # Try semantic first (KeyBERT), fallback to YAKE, then statistical
    keywords = extract_keywords_keybert(text, top_n=5)
    if not keywords:
        keywords = extract_keywords_yake(text, top_n=5)
    if not keywords:
        keywords = extract_keywords(text, top_n=5, df_map=df_map, num_docs=num_docs)
    return {'keywords': keywords, 'entities': ents, 'cta': _extract_cta(text), 'intent': classify_intent(text)}


def analyze_text_for_video(video_id: str, transcription: str, df_map: Dict[str, int] | None = None, num_docs: int | None = None) -> Dict[str, object]:
    return analyze_text(transcription, df_map=df_map, num_docs=num_docs)


def analyze_text_batch(video_ids: List[str], transcriptions: Dict[str, dict]) -> Dict[str, dict]:
    texts = []
    id_to_text: Dict[str, str] = {}
    for vid in video_ids:
        tdata = transcriptions.get(vid, {})
        txt = tdata.get('transcription', '') if isinstance(tdata, dict) else (str(tdata) if tdata else '')
        id_to_text[vid] = txt
        if txt and txt.strip():
            texts.append(txt)
    df_map = _build_df_map(texts)
    num_docs = len(texts) if texts else 0
    return {vid: analyze_text(id_to_text.get(vid, ''), df_map=df_map, num_docs=num_docs) for vid in video_ids}

