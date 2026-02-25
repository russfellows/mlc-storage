"""
Stateful multi-turn conversation management for KV Cache Benchmark.

Tracks conversation state and cache key history across turns,
enabling cache reuse in conversational AI workloads.
"""

import time
import hashlib
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from kv_cache.config import cfg
from kv_cache.models import InferenceRequest


@dataclass
class ConversationState:
    """Tracks the state of a single multi-turn conversation for a user."""
    conversation_id: str
    user_id: str
    turn_number: int
    created_at: datetime
    last_access: datetime

    # KV cache management for this conversation.
    cache_keys: List[str] = field(default_factory=list)
    cumulative_tokens: int = 0
    cache_locations: Dict[str, str] = field(default_factory=dict)

    # Metadata for advanced caching strategies.
    system_prompt_key: Optional[str] = None
    common_prefix_keys: List[str] = field(default_factory=list)

    # Performance tracking for this conversation.
    turns_completed: int = 0
    total_latency: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0


class ConversationManager:
    """Manages the lifecycle of all multi-turn conversations and enables cache reuse."""

    def __init__(self, max_conversations: int = None, max_turns_per_conv: int = None):
        self.conversations: Dict[str, ConversationState] = {}
        self.max_conversations = max_conversations if max_conversations is not None else cfg('conversation', 'max_conversations', default=1000)
        self.max_turns_per_conv = max_turns_per_conv if max_turns_per_conv is not None else cfg('conversation', 'max_turns_per_conv', default=50)
        self.lock = threading.Lock()

    def start_conversation(self, user_id: str, system_prompt: Optional[str] = None) -> str:
        """Initializes a new conversation for a given user."""
        conv_id = f"conv_{user_id}_{int(time.time()*1000)}"

        state = ConversationState(
            conversation_id=conv_id,
            user_id=user_id,
            turn_number=0,
            created_at=datetime.now(),
            last_access=datetime.now(),
            cache_keys=[],
            cumulative_tokens=0,
            cache_locations={}
        )

        if system_prompt:
            state.system_prompt_key = f"system_prompt_{hashlib.sha256(system_prompt.encode()).hexdigest()[:16]}"

        with self.lock:
            if len(self.conversations) >= self.max_conversations:
                self._evict_oldest_conversation()

            self.conversations[conv_id] = state

        return conv_id

    def add_turn(self, conversation_id: str, user_message_tokens: int,
                 assistant_response_tokens: int) -> Tuple[int, str]:
        """Adds a new turn to an existing conversation, updating its state."""
        with self.lock:
            if conversation_id not in self.conversations:
                raise ValueError(f"Conversation {conversation_id} not found")

            state = self.conversations[conversation_id]
            state.turn_number += 1
            state.last_access = datetime.now()

            turn_cache_key = f"{conversation_id}_turn_{state.turn_number}"

            state.cache_keys.append(turn_cache_key)
            state.cumulative_tokens += user_message_tokens + assistant_response_tokens
            state.turns_completed += 1

            return state.turn_number, turn_cache_key

    def get_conversation_context_size(self, conversation_id: str) -> int:
        """Gets the total number of tokens accumulated in a conversation."""
        with self.lock:
            if conversation_id not in self.conversations:
                return 0
            return self.conversations[conversation_id].cumulative_tokens

    def get_all_previous_turn_keys(self, conversation_id: str, current_turn: int) -> List[str]:
        """Retrieves all cache keys from previous turns in a conversation."""
        with self.lock:
            if conversation_id not in self.conversations:
                return []
            state = self.conversations[conversation_id]
            return [key for key in state.cache_keys if key != f"{conversation_id}_turn_{current_turn}"]

    def _evict_oldest_conversation(self):
        """Evicts the least recently used (LRU) conversation to make space."""
        if not self.conversations:
            return
        oldest_conv_id = min(
            self.conversations,
            key=lambda k: (self.conversations[k].last_access, self.conversations[k].created_at)
        )
        del self.conversations[oldest_conv_id]
