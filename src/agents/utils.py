import re
from typing import List
from langchain_core.messages import BaseMessage, SystemMessage

def trim_messages(messages: List[BaseMessage], max_messages: int = 15) -> List[BaseMessage]:
    """
    Trims the message history to keep the context window small for local LLMs.
    Always preserves the first SystemMessage if present.
    """
    if len(messages) <= max_messages:
        return messages

    system_msg = None
    if messages and isinstance(messages[0], SystemMessage):
        system_msg = messages[0]
        remaining_messages = messages[1:]
    else:
        remaining_messages = messages

    trimmed = remaining_messages[-max_messages:]
    
    if system_msg:
        return [system_msg] + trimmed
    return trimmed

def mini_rag_filter(content: str, query: str, max_chars: int = 5000) -> str:
    """
    Semi-intelligent 'Mini-RAG' filter.
    Instead of hard truncation, it extracts the most relevant paragraphs based on query overlap.
    """
    if len(content) <= max_chars:
        return content

    # 1. Clean and tokenize the query for scoring
    # We remove common stop words and keep meaningful keywords
    keywords = set(re.findall(r'\w+', query.lower()))
    stop_words = {'the', 'a', 'is', 'of', 'and', 'to', 'in', 'for', 'with', 'on', 'give', 'me', 'what', 'search', 'find'}
    keywords = keywords - stop_words

    # 2. Split content into paragraphs/sections
    paragraphs = content.split('\n\n')
    scored_paragraphs = []

    for p in paragraphs:
        if not p.strip():
            continue
        # Score based on keyword overlap
        score = sum(1 for word in keywords if word in p.lower())
        scored_paragraphs.append((score, p))

    # 3. Sort by score (descending)
    scored_paragraphs.sort(key=lambda x: x[0], reverse=True)

    # 4. Reconstruct content within the budget
    final_output = []
    current_size = 0
    
    # We always prioritize the first paragraph if it's high quality (often contains the intro/summary)
    # But for search results, we just follow the score.
    
    for score, p in scored_paragraphs:
        if current_size + len(p) <= max_chars:
            final_output.append(p)
            current_size += len(p)
        elif not final_output: # Even if first paragraph is too big, take a slice of it
            final_output.append(p[:max_chars])
            break
        else:
            break

    if not final_output:
        return content[:max_chars] + "\n\n[Truncated]"

    result = "\n\n".join(final_output)
    if len(result) < len(content):
        result += f"\n\n... [Focused context extracted. Original size: {len(content)} chars] ..."
    
    return result

def truncate_tool_output(content: str, max_chars: int = 5000) -> str:
    """Fallback for when no query is available for filtering."""
    if len(content) <= max_chars:
        return content
    return content[:max_chars] + f"\n\n... [Truncated for brevity] ..."