"""Implement DQ-PCA, ranking-only LoRA and task-specific feed-forward paths here.

Define the stop-gradient boundaries explicitly; shared sparse embeddings may have
different gradient rules from the base attention projections.
"""
