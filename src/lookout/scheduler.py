"""Priority scheduler: the reason this project exists.

Design (settled — see CLAUDE.md):

- One priority queue per model name. Higher priority number wins; FIFO within a
  priority. An InferenceJob carries (model, payload_ref, priority, chain_id,
  step_id, enqueued_at).
- A Worker is configured with the list of models it can serve (v1: one worker,
  all models, since everything runs on the one MI50). It repeatedly picks the
  highest-priority job across its models' queues and runs it. This is what makes
  "a second box that only hosts qwen3-vl" a config entry later, not a redesign.
- No mid-inference preemption: a gesture job at priority 100 jumps every queue
  but never cancels an in-flight call.
- Every inference call is wrapped in the metrics timing context manager
  (metrics.py) — labels {model, tier}.

Implement with stdlib: heapq or queue.PriorityQueue keyed by (-priority,
enqueued_at). No async framework; threads are fine at this scale.
"""
