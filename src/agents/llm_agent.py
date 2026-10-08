"""
LLM agent using OpenRouter API for decision making.
Uses configurable prompt styles for different experimental conditions.

Design references:
- Reasoning traces as data: thinking tokens (<think>...</think>) are extracted
  and stored as behavioral data, not as mechanistic explanations. Faithfulness
  caveats per Turpin et al. (2023), Lanham et al. (2023), Chen et al. (2025).
- Thinking model support: Qwen3/3.5 reasoning via enable_thinking chat template.
  vLLM reasoning parser separates thinking from content tokens.
- Retry with structured output: JSON follow-up prompt on parse failure,
  preserving the original reasoning trace for analysis.
"""

import os
import json
import re
import time
import sys
from typing import Dict, Optional
from openai import OpenAI
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from game.engine import Action, ActionType
from agents.prompts import get_prompt_style
from agents.memory import AgentMemory


class LLMAgent:
    """LLM-based agent that makes decisions via OpenRouter API."""

    def __init__(self,
                 agent_id: str,
                 api_key: str,
                 model: str,
                 prompt_config: Optional[Dict] = None,
                 game_params: Optional[Dict] = None,
                 temperature: float = 0.7,
                 max_tokens: int = 500,
                 timeout: int = 30,
                 retry_attempts: int = 3,
                 retry_delay: int = 2,
                 base_url: str = "https://openrouter.ai/api/v1",
                 memory_config: Optional[Dict] = None):
        """
        Initialize LLM agent.

        Args:
            agent_id: Unique identifier for this agent
            api_key: OpenRouter API key (use "none" for local vLLM)
            model: Model identifier (e.g., "deepseek/deepseek-v3.2")
            prompt_config: Dictionary with prompt toggles (objective_style, state_style, etc.)
            game_params: Game parameters dict
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            timeout: Request timeout in seconds
            retry_attempts: Number of retry attempts on failure
            retry_delay: Delay between retries in seconds
            base_url: API base URL (OpenRouter, vLLM local, etc.)
            memory_config: Memory settings dict with 'enabled', 'window_size',
                'notes_persist' (notes kept whole game vs windowed), etc.
        """
        self.agent_id = agent_id
        self.model = model
        self.temperature = temperature
        # Thinking models need more tokens for reasoning + JSON response.
        # Qwen3/3.5 and Gemma 4 thinking tokens count toward completion_tokens —
        # 2048 is far too low (thinking alone uses 4-6K tokens). Floor at 4096
        # so the output budget leaves room for prompt growth in late rounds.
        model_lower = model.lower()
        self.is_thinking_model = any(
            t in model_lower for t in ["qwq", "qwen3", "gemma4", "gemma-4"]
        )
        if self.is_thinking_model and max_tokens < 4096:
            self.max_tokens = 4096
        else:
            self.max_tokens = max_tokens
        # Thinking models need longer timeout (long reasoning chains)
        # With 8+ concurrent requests, per-request throughput drops to ~30 tok/s.
        # 6K thinking tokens / 30 tok/s = 200s decode + prefill + scheduling.
        # Qwen3.5 vLLM recipe uses 3600s. 900s is a safe practical minimum.
        if self.is_thinking_model and timeout < 900:
            self.timeout = 900
        else:
            self.timeout = timeout
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay

        # Initialize prompt
        self.game_params = game_params or {}
        self.prompt = get_prompt_style(prompt_config or {}, game_params)

        # Initialize memory
        mem_cfg = memory_config or {}
        self.memory_enabled = mem_cfg.get('enabled', True)
        if self.memory_enabled:
            window_size = mem_cfg.get('window_size', 10)
            notes_persist = mem_cfg.get('notes_persist', True)
            self.memory = AgentMemory(agent_id, window_size=window_size,
                                      notes_persist=notes_persist)
        else:
            self.memory = None

        # Initialize OpenAI-compatible client (works with OpenRouter, vLLM, etc.)
        self.is_openrouter = "openrouter" in (base_url or "")
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        
        self.reasoning_traces = []
        self._visible_agents = None
        self._last_message = None  # Last message extracted from LLM response
        self._last_memory = None   # Last free-form memory entry extracted from LLM response
        self._last_rewire_nom = None  # {'drop': id_or_None, 'invite': id_or_None}
    
    def _format_observation(self, observation: Dict) -> str:
        """
        Format observation using unified prompt.
        """
        return self.prompt.format_observation(observation, self.agent_id)

    def _retry_schema_fields(self) -> list:
        """Schema fields the JSON-retry follow-up must request, gated the same way
        as the main prompt template (I4). Without this, the retry omitted
        harvest/message/rewire and silently zeroed them on a parse-retry."""
        gp = self.game_params
        fields = ['"reasoning": "<think step by step>"',
                  '"action": "<action_name>"',
                  '"target": "<agent_id or null>"']
        if self.prompt.comm_scope != "none":
            fields.append('"message": "<your message, or null>"')
            fields.append('"message_to": "<list of recipient agent_ids, or null>"')
        if gp.get('commons_enabled', False):
            fields.append('"harvest": "<how much to take from the shared stock, or 0>"')
        return fields
    
    def _parse_response(self, response_text: str) -> Optional[Dict]:
        """
        Parse LLM response to extract action.

        Handles <think>...</think> blocks from reasoning models (Qwen3, QwQ, MiMo):
        extracts thinking content separately, then strips it before JSON parsing.

        Fallbacks:
        - Strips markdown code fences (```json ... ```) before JSON extraction
        - Strips trailing commas before closing braces (common LLM error)
        - Fuzzy action name matching (e.g. "invest_Others" → "invest_other")

        Returns:
            Dictionary with action, target, reasoning, and thinking, or None if parsing fails
        """
        try:
            # Extract <think> content before stripping
            think_match = re.search(r'<think>(.*?)</think>', response_text, flags=re.DOTALL)
            thinking_content = think_match.group(1).strip() if think_match else None

            # Strip <think>...</think> blocks before JSON extraction
            clean_text = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL)

            # Strip markdown code fences (```json ... ``` or ``` ... ```)
            clean_text = re.sub(r'```(?:json)?\s*', '', clean_text)

            # Try to find JSON in cleaned response
            start_idx = clean_text.find('{')
            end_idx = clean_text.rfind('}') + 1

            if start_idx >= 0 and end_idx > start_idx:
                json_str = clean_text[start_idx:end_idx]

                # Fix trailing commas before closing braces (common LLM error)
                json_str = re.sub(r',\s*}', '}', json_str)
                json_str = re.sub(r',\s*]', ']', json_str)

                parsed = json.loads(json_str)

                # Validate required fields
                if 'action' in parsed:
                    # Normalize action name (fuzzy match)
                    action = self._normalize_action(parsed['action'])
                    if action is None:
                        return None

                    result = {
                        'action': action,
                        'target': parsed.get('target'),
                        'thinking': thinking_content,
                    }
                    # Extract communication fields if present
                    if parsed.get('message'):
                        result['message'] = parsed['message']
                        result['message_to'] = parsed.get('message_to')
                    # Extract free-form memory entry if present
                    if parsed.get('memory'):
                        result['memory'] = parsed['memory']
                    # Extract commons harvest claim if present
                    if parsed.get('harvest') is not None:
                        result['harvest'] = parsed.get('harvest')
                    return result
        except json.JSONDecodeError:
            pass

        return None

    # Canonical action vocabulary. 'harvest' is only a real action under
    # commons_harvest_mode="action_own" (gated in _action_dict_to_action);
    # in other modes a stray 'harvest' string snaps to hold.
    _VALID_ACTIONS = {
        'transfer', 'strengthen', 'take', 'hold', 'harvest',
        # rewiring as an action (production behaviour; gated on assoc_enabled
        # in _action_dict_to_action)
        'drop', 'invite',
    }

    # Legacy names + shorthands → canonical (input backward compatibility)
    _ACTION_ALIASES = {
        'invest_other': 'transfer', 'invest': 'transfer', 'transfer_other': 'transfer',
        'arm_other': 'strengthen', 'arm': 'strengthen', 'strengthen_other': 'strengthen',
        'attack': 'take',
        'do_nothing': 'hold', 'donothing': 'hold', 'nothing': 'hold',
    }

    def _normalize_action(self, action_raw: str) -> Optional[str]:
        """Normalize an action string to the canonical vocabulary
        (transfer / strengthen / take / hold). Accepts legacy names
        (invest_other / arm_other / attack / do_nothing), case, spaces,
        hyphens, and a trailing 's'.
        """
        action = (action_raw or '').strip().lower().replace(' ', '_').replace('-', '_')

        if action in self._VALID_ACTIONS:
            return action
        if action.endswith('s') and action[:-1] in self._VALID_ACTIONS:
            return action[:-1]
        if action in self._ACTION_ALIASES:
            return self._ACTION_ALIASES[action]
        return None

    def _action_dict_to_action(self, action_dict: Dict) -> Optional[Action]:
        """Convert parsed action dictionary to Action object."""
        action_str = self._normalize_action(action_dict.get('action', ''))
        target = action_dict['target']

        # Map canonical action string to ActionType
        action_map = {
            'transfer': ActionType.INVEST_OTHER,
            'strengthen': ActionType.ARM_OTHER,
            'take': ActionType.ATTACK,
            'hold': ActionType.DO_NOTHING,
            'harvest': ActionType.HARVEST,
        }

        # Rewiring is an ACTION (production behaviour):
        # drop/invite spend the round on the network move. Economically neutral;
        # the nomination is stored here and picked up by the runner->network path.
        if self.game_params.get('assoc_enabled', True) and action_str in ('drop', 'invite'):
            if not target or target == self.agent_id:
                return None
            if action_str == 'drop' and self._visible_agents is not None \
                    and target not in self._visible_agents:
                return None  # can only drop a current neighbour
            self._last_rewire_nom = ({'drop': target, 'invite': None}
                                     if action_str == 'drop'
                                     else {'drop': None, 'invite': target})
            return Action(agent_id=self.agent_id,
                          action_type=ActionType.DO_NOTHING, target_id=None)

        if not action_str or action_str not in action_map:
            return None

        action_type = action_map[action_str]

        # Disabled affordances snap to hold — consistent with how the engine
        # neutralises them and how harvest snaps to 0 when commons is off.
        if action_type == ActionType.ARM_OTHER and not self.game_params.get('arm_enabled', True):
            action_type = ActionType.DO_NOTHING
            target = None
        if action_type == ActionType.ATTACK and not self.game_params.get('take_enabled', True):
            action_type = ActionType.DO_NOTHING
            target = None
        # HARVEST exists only on the commons rung.
        if action_type == ActionType.HARVEST and not self.game_params.get('commons_enabled', False):
            action_type = ActionType.DO_NOTHING
            target = None

        # Validate target requirements
        if action_type in [ActionType.INVEST_OTHER, ActionType.ARM_OTHER, ActionType.ATTACK]:
            if not target or target == self.agent_id:
                return None
            # Network restriction: actions only allowed on neighbors
            if self._visible_agents is not None and target not in self._visible_agents:
                return None

        raw_harvest = action_dict.get('harvest', 0)
        return Action(
            agent_id=self.agent_id,
            action_type=action_type,
            target_id=target if target else None,
            harvest=self._snap_harvest(raw_harvest),
            harvest_raw=str(raw_harvest) if raw_harvest not in (None, 0, '0') else None,
        )

    def _snap_harvest(self, raw) -> float:
        """Parse the model's harvest claim as a continuous, unbounded,
        non-negative PERCENT of the agent's own resources, returned as a
        FRACTION. Convention: the field is a percent number, so "5", "5%", and
        "0.05" are all read as the NUMBER first, then divided by 100 -> 0.05.
        No upper bound; negatives clamp to 0. Returns 0.0 if commons is off or
        the value is unusable."""
        if not self.game_params.get('commons_enabled', False):
            return 0.0

        had_percent = isinstance(raw, str) and '%' in raw
        if isinstance(raw, str):
            raw = raw.strip().rstrip('%').strip()
        try:
            v = float(raw)
        except (TypeError, ValueError):
            return 0.0
        if v <= 0:
            return 0.0
        # Audit flag (C2): a bare decimal < 1 with no '%' is ambiguous — it
        # could be an intended fraction (0.05) rather than a percent (0.05% →
        # 0.0005). We DO NOT rescale (percent semantics are fixed); just warn
        # so any mis-scale is detectable post-hoc.
        if v < 1 and not had_percent:
            print(f"WARN: {getattr(self, 'agent_id', '?')} harvest '{raw}' is a "
                  f"bare decimal < 1 (read as {v}% -> fraction {v / 100.0}); verify intent.")
        return v / 100.0
    
    def _store_message(self, action_dict: Dict):
        """Store communication message from LLM response.

        message_to is normalized to a LIST of target agent_ids. There is no
        broadcast channel and no "all" keyword: to reach many agents the model
        lists many ids. Self, blanks, and the legacy "all"/"null" tokens are
        dropped; addressing is otherwise free (delivery is gated to living
        agents downstream in the runner).
        """
        msg_text = action_dict.get('message', '')
        if not (msg_text and msg_text.strip()):
            return
        raw_to = action_dict.get('message_to')
        if isinstance(raw_to, str):
            raw_to = [raw_to]
        targets: list = []
        if isinstance(raw_to, (list, tuple)):
            for t in raw_to:
                if not isinstance(t, str):
                    continue
                t = t.strip()
                if not t or t.lower() in ('all', 'null', 'none'):
                    continue
                if t == self.agent_id or t in targets:
                    continue
                targets.append(t)
        if not targets:
            return
        self._last_message = {
            'from': self.agent_id,
            'message': msg_text.strip(),
            'message_to': targets,
        }

    def _store_memory(self, action_dict: Dict):
        """Store free-form memory entry from LLM response."""
        mem = action_dict.get('memory')
        if mem and isinstance(mem, str) and mem.strip():
            self._last_memory = mem.strip()
        else:
            self._last_memory = None

    def get_last_message(self) -> Optional[Dict]:
        """Return the message from the last select_action call, or None."""
        return self._last_message

    def get_last_rewire_nomination(self) -> Optional[Dict]:
        """Return the rewire nomination from the last select_action call, or None."""
        return self._last_rewire_nom

    def select_action(self, observation: Dict) -> Action:
        """
        Select action based on observation using LLM.
        
        Args:
            observation: Game state observation
            
        Returns:
            Selected action
        """
        # Store visible agents for network target validation
        self._visible_agents = observation.get('visible_agents', None)
        self._last_message = None  # Reset each round
        self._last_memory = None   # Reset each round
        self._last_rewire_nom = None  # Reset each round

        prompt = self._format_observation(observation)

        errors = []

        for attempt in range(self.retry_attempts):
            try:
                t0 = time.time()
                # Build API call kwargs
                api_kwargs = dict(
                    model=self.model,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    timeout=self.timeout,
                )
                # Qwen3 / Gemma 4: enable thinking via chat template kwarg.
                # vLLM exposes the chain-of-thought through reasoning_content
                # when the matching --reasoning-parser is enabled server-side.
                if self.is_thinking_model:
                    if self.is_openrouter:
                        # OpenRouter only RETURNS the chain-of-thought when asked:
                        # without this the reasoning tokens are billed but the
                        # message.reasoning field stays absent (traces lost).
                        api_kwargs["extra_body"] = {"reasoning": {"enabled": True}}
                    else:
                        api_kwargs["extra_body"] = {
                            "chat_template_kwargs": {"enable_thinking": True}
                        }

                response = self.client.chat.completions.create(**api_kwargs)
                latency = time.time() - t0

                msg = response.choices[0].message
                response_text = msg.content or ""

                # Extract thinking from vLLM --reasoning-parser output
                # vLLM uses 'reasoning' attr (not 'reasoning_content')
                thinking_content = getattr(msg, 'reasoning_content', None) or getattr(msg, 'reasoning', None)

                # Extract token usage from response
                usage = {}
                if response.usage:
                    usage = {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens,
                    }

                # Parse response (extracts thinking + action)
                action_dict = self._parse_response(response_text)

                # If content was empty but reasoning_content has JSON, try extracting from there
                if not action_dict and thinking_content:
                    action_dict = self._parse_response(thinking_content)

                # Attach reasoning_content as thinking data
                if thinking_content and action_dict:
                    action_dict['thinking'] = thinking_content

                # Log reasoning trace (includes thinking if present)
                trace_entry = {
                    "round": observation['round'],
                    "agent_id": self.agent_id,
                    "prompt": prompt,
                    "response": response_text,
                    "model": self.model,
                    "latency_s": round(latency, 3),
                    "usage": usage,
                    "attempt": attempt + 1,
                    "errors": errors.copy(),
                }
                final_thinking = thinking_content or (action_dict.get('thinking') if action_dict else None)
                if final_thinking is not None:
                    trace_entry["thinking"] = final_thinking
                self.reasoning_traces.append(trace_entry)

                if action_dict:
                    action = self._action_dict_to_action(action_dict)
                    if action:
                        self._store_message(action_dict)
                        self._store_memory(action_dict)
                        return action

                # JSON missing — send a follow-up requesting structured output
                # Use low max_tokens for retry: only need JSON, not reasoning
                if attempt < self.retry_attempts - 1:
                    try:
                        t0 = time.time()
                        retry_format = "{" + ", ".join(self._retry_schema_fields()) + "}"
                        retry_response = self.client.chat.completions.create(
                            model=self.model,
                            messages=[
                                {"role": "user", "content": prompt},
                                {"role": "assistant", "content": response_text},
                                {"role": "user", "content": f"Please respond with ONLY a JSON object in this exact format: {retry_format}"}
                            ],
                            temperature=self.temperature,
                            max_tokens=512,
                            timeout=60
                        )
                        retry_latency = time.time() - t0
                        retry_text = retry_response.choices[0].message.content

                        retry_usage = {}
                        if retry_response.usage:
                            retry_usage = {
                                "prompt_tokens": retry_response.usage.prompt_tokens,
                                "completion_tokens": retry_response.usage.completion_tokens,
                                "total_tokens": retry_response.usage.total_tokens,
                            }

                        action_dict = self._parse_response(retry_text)
                        if action_dict:
                            # Audit (I4): warn if a schema field that the original
                            # response carried got dropped by the retry → silently zeroed.
                            for f in ("harvest", "message", "message_to"):
                                if f in (response_text or "") and f not in action_dict:
                                    print(f"WARN: {self.agent_id} JSON-retry dropped "
                                          f"previously-present field '{f}' (zeroed).")
                            action = self._action_dict_to_action(action_dict)
                            if action:
                                self._store_message(action_dict)
                                self._store_memory(action_dict)
                                # Log the retry trace
                                self.reasoning_traces.append({
                                    "round": observation['round'],
                                    "agent_id": self.agent_id,
                                    "prompt": "(JSON retry)",
                                    "response": retry_text,
                                    "model": self.model,
                                    "latency_s": round(retry_latency, 3),
                                    "usage": retry_usage,
                                    "attempt": attempt + 1,
                                    "is_retry": True,
                                })
                                return action
                    except Exception:
                        pass
                    time.sleep(self.retry_delay)
                    continue

            except Exception as e:
                errors.append({"attempt": attempt + 1, "error": str(e)})
                print(f"Error in LLM request (attempt {attempt + 1}/{self.retry_attempts}): {e}")
                if attempt < self.retry_attempts - 1:
                    time.sleep(self.retry_delay)

        # Try to recover intended action from thinking traces before giving up.
        # When thinking models exhaust their token budget, the thinking often
        # contains the decision ("Attack Storm", "invest in Sage") even though
        # the JSON response was never generated.
        last_trace = self.reasoning_traces[-1] if self.reasoning_traces else None
        thinking_text = last_trace.get("thinking", "") if last_trace else ""
        fallback_action = self._parse_action_from_thinking(thinking_text, observation)

        if fallback_action:
            print(f"Warning: {self.agent_id} recovered action from thinking: "
                  f"{fallback_action.action_type.name} -> {fallback_action.target_id}")
            # Log as thinking-recovered fallback
            self.reasoning_traces.append({
                "round": observation['round'],
                "agent_id": self.agent_id,
                "prompt": prompt,
                "response": None,
                "model": self.model,
                "fallback": "thinking_recovery",
                "recovered_action": fallback_action.action_type.name.lower(),
                "recovered_target": fallback_action.target_id,
                "errors": errors,
            })
            return fallback_action

        # Last resort: do_nothing (safest neutral action)
        print(f"Warning: {self.agent_id} falling back to do_nothing (no action recoverable)")
        self.reasoning_traces.append({
            "round": observation['round'],
            "agent_id": self.agent_id,
            "prompt": prompt,
            "response": None,
            "model": self.model,
            "fallback": "default",
            "errors": errors,
        })
        return Action(
            agent_id=self.agent_id,
            action_type=ActionType.DO_NOTHING,
            target_id=None
        )
    
    def _parse_action_from_thinking(self, thinking: str, observation: Dict) -> Optional[Action]:
        """Try to recover intended action from thinking text when JSON response is missing.

        When thinking models exhaust their token budget, the thinking often contains
        the intended decision even though the JSON was never generated. This method
        scans for decision patterns in the thinking text.

        Strategy: scan the TAIL (last 2000 chars) for explicit decision statements,
        then fall back to the LAST mention of an action + target pair anywhere in
        the thinking. The last mention is most likely the final decision.

        Returns Action if a clear decision is found, None otherwise.
        """
        if not thinking or len(thinking) < 50:
            return None

        # Known agent IDs from observation
        all_agents = set(observation.get('resources', {}).keys())
        all_agents.discard(self.agent_id)
        # Build case-insensitive lookup
        agent_lookup = {a.lower(): a for a in all_agents}

        def _match_agent(name: str) -> Optional[str]:
            """Match a candidate name to a known agent ID."""
            return agent_lookup.get(name.lower())

        def _try_build_action(action_name: str, target_name: Optional[str]) -> Optional[Action]:
            """Try to build a valid Action from an action name and optional target."""
            action_str = self._normalize_action(action_name)
            if not action_str:
                return None
            action_map = {
                'transfer': ActionType.INVEST_OTHER,
                'strengthen': ActionType.ARM_OTHER,
                'take': ActionType.ATTACK,
                'hold': ActionType.DO_NOTHING,
                'harvest': ActionType.HARVEST,
            }
            action_type = action_map.get(action_str)
            if not action_type:
                return None
            needs_target = action_type in (ActionType.INVEST_OTHER, ActionType.ARM_OTHER, ActionType.ATTACK)
            target_id = _match_agent(target_name) if target_name else None
            if needs_target and not target_id:
                return None
            return Action(agent_id=self.agent_id, action_type=action_type,
                          target_id=target_id if needs_target else None)

        tail = thinking[-2000:]
        tail_lower = tail.lower()

        # --- Phase 1: Explicit decision statements (high confidence) ---
        decision_prefixes = (
            r'(?:decision|final plan|final choice|final confirmation|my (?:best )?action|'
            r'plan confirmed|okay,? (?:ready|proceeding)|i\'m confident)'
        )
        # "Decision: Attack Storm" / "Final Plan: invest_other -> Sage"
        for pattern in [
            decision_prefixes + r'[:\s]*\**\s*(attack)\s+(\w+)',
            decision_prefixes + r'[:\s]*\**\s*(invest(?:[_ ]?other)?(?:[_ ]?in)?)\s+(\w+)',
            decision_prefixes + r'[:\s]*\**\s*(arm[_ ]?other)\s+(\w+)',
            decision_prefixes + r'[:\s]*\**\s*(arm[_ ]?self)',
            decision_prefixes + r'[:\s]*\**\s*(do[_ ]?nothing)',
        ]:
            match = re.search(pattern, tail_lower)
            if match:
                groups = match.groups()
                action_name = groups[0].replace(' ', '_').strip('_')
                target_name = groups[1] if len(groups) > 1 else None
                # Normalize "invest in" / "invest" to "invest_other" when target present
                if action_name.startswith('invest') and target_name and 'self' not in action_name:
                    action_name = 'invest_other'
                result = _try_build_action(action_name, target_name)
                if result:
                    return result

        # --- Phase 2: "Action: attack. Target: Storm." pattern ---
        action_target_match = re.search(
            r'action[:\s]*\**\s*(attack|invest[_ ]?other|arm[_ ]?(?:self|other)|do[_ ]?nothing)\**'
            r'.*?target[:\s]*\**\s*(\w+)',
            tail_lower, re.DOTALL
        )
        if action_target_match:
            action_name = action_target_match.group(1).replace(' ', '_')
            target_name = action_target_match.group(2)
            if target_name in ('null', 'none', 'n/a'):
                target_name = None
            result = _try_build_action(action_name, target_name)
            if result:
                return result

        # --- Phase 3: Last mention of action + agent name (lower confidence) ---
        # Scan for the LAST occurrence of "attack <Agent>" etc. in the tail
        agent_pattern = '|'.join(re.escape(a) for a in agent_lookup.keys())
        last_action_patterns = [
            (rf'(take|attack)\s+({agent_pattern})', 'take'),
            (rf'(transfer|invest(?:[_ ]?(?:other|in))?)\s+({agent_pattern})', 'transfer'),
            (rf'(strengthen|arm[_ ]?other)\s+({agent_pattern})', 'strengthen'),
            (rf'(hold|do[_ ]?nothing)', 'hold'),
        ]
        best_match = None
        best_pos = -1
        for pattern, action_name in last_action_patterns:
            for match in re.finditer(pattern, tail_lower):
                if match.start() > best_pos:
                    groups = match.groups()
                    target_name = groups[1] if len(groups) > 1 else None
                    result = _try_build_action(action_name, target_name)
                    if result:
                        best_match = result
                        best_pos = match.start()

        return best_match

    def update_memory(self, round_num: int, action_str: str, target: Optional[str],
                      outcome: Optional[Dict], visible_agents: Optional[list],
                      round_actions: list,
                      all_resources: Optional[Dict] = None,
                      received_messages: Optional[list] = None,
                      rewire: Optional[Dict] = None):
        """Update agent memory after a round resolves.

        Args:
            round_num: Round number just resolved.
            action_str: This agent's action type string.
            target: Target agent ID or None.
            outcome: Dict with outcome details (e.g. resource_change, combat_won).
            visible_agents: List of visible agent IDs (None = all).
            round_actions: All actions from the round log.
            all_resources: Post-round resources for all agents. Memory stores
                           only self + visible in the per-round snapshot.
            received_messages: Messages received this round (list of dicts).
            rewire: Agent's rewiring nomination + outcome for this round, or None.
        """
        if not self.memory_enabled or self.memory is None:
            return
        own_action = {
            'action': action_str,
            'target': target,
            'outcome': outcome or {},
        }
        self.memory.record_round(
            round_num=round_num,
            own_action=own_action,
            round_actions=round_actions,
            visible_agents=visible_agents,
            all_resources=all_resources,
            sent_message=self._last_message,
            received_messages=received_messages or [],
            rewire=rewire,
        )
        self.memory.record_memory(round_num, self._last_memory)

    def get_reasoning_traces(self, include_retries=False) -> list:
        """Get reasoning traces for analysis.

        By default excludes JSON-retry entries so trace count == round count.
        Set include_retries=True to get all entries (for debugging).
        """
        if include_retries:
            return self.reasoning_traces
        return [t for t in self.reasoning_traces if not t.get('is_retry')]
