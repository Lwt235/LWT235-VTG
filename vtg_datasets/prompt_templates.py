"""
Prompt Templates for Video Temporal Localization.

This module provides a collection of diverse prompt templates for temporal
grounding tasks to increase training variety and model robustness.
"""

from typing import List
import random


# Templates for standard (non-temporal-token) mode
STANDARD_TEMPLATES = [
    # Direct instruction style
    "Given the video, please identify the start and end time of the moment described by the following query: \"{query}\"\nProvide the answer in the format: [start_time, end_time]",
    
    # Question style
    "In the provided video, when does the following event occur: \"{query}\"?\nPlease specify the start and end timestamps.",
    
    # Task description style
    "Watch the video and locate the temporal segment where: \"{query}\"\nOutput the start and end times of this segment.",
    
    # Natural language style
    "Please analyze the video and tell me the time range when the following happens: \"{query}\"\nProvide your answer as start and end times.",
    
    # Concise style
    "Find when \"{query}\" occurs in the video.\nAnswer with: [start, end]",
    
    # Detailed style
    "Your task is to watch the video carefully and identify the exact temporal boundaries of the moment described as: \"{query}\"\nReport the start time and end time.",
    
    # Academic style
    "Given a video sequence, determine the temporal interval [t_start, t_end] corresponding to the query: \"{query}\"",
    
    # Conversational style
    "I need to find a specific moment in this video. The description is: \"{query}\"\nCan you tell me when it starts and when it ends?",
]


# Templates for temporal token mode
TEMPORAL_TOKEN_TEMPLATES = [
    # Direct instruction style
    "Given the video, please identify the start and end time of the moment described by the following query: \"{query}\"\nProvide the answer using temporal tokens in the format: [start, end]",
    
    # Question style
    "In the provided video, when does the following event occur: \"{query}\"?\nPlease specify using temporal tokens.",
    
    # Task description style
    "Watch the video and locate the temporal segment where: \"{query}\"\nOutput using temporal tokens to indicate the start and end.",
    
    # Natural language style
    "Please analyze the video and tell me when the following happens: \"{query}\"\nUse temporal tokens to mark the time boundaries.",
    
    # Concise style
    "Find when \"{query}\" occurs in the video.\nAnswer with temporal tokens: [start, end]",
    
    # Detailed style
    "Your task is to watch the video carefully and identify the exact temporal boundaries of the moment described as: \"{query}\"\nUse temporal tokens to indicate the start and end times.",
    
    # Academic style
    "Given a video sequence, determine the temporal interval using temporal tokens for the query: \"{query}\"",
    
    # Conversational style
    "I need to find a specific moment in this video. The description is: \"{query}\"\nCan you mark the start and end using temporal tokens?",
]


def get_prompt_templates(use_temporal_tokens: bool = False) -> List[str]:
    """
    Get the list of available prompt templates.
    
    Args:
        use_temporal_tokens: Whether to get templates for temporal token mode.
    
    Returns:
        List of prompt template strings.
    """
    if use_temporal_tokens:
        return TEMPORAL_TOKEN_TEMPLATES.copy()
    return STANDARD_TEMPLATES.copy()


def get_random_template(use_temporal_tokens: bool = False, seed: int = None) -> str:
    """
    Get a random prompt template.
    
    Args:
        use_temporal_tokens: Whether to get template for temporal token mode.
        seed: Optional random seed for reproducibility.
    
    Returns:
        A randomly selected prompt template string.
    """
    templates = get_prompt_templates(use_temporal_tokens)
    
    if seed is not None:
        rng = random.Random(seed)
        return rng.choice(templates)
    
    return random.choice(templates)


def format_prompt(template: str, query: str) -> str:
    """
    Format a prompt template with a query.
    
    Args:
        template: Template string with {query} placeholder.
        query: The query text to insert.
    
    Returns:
        Formatted prompt string.
    """
    return template.format(query=query)


class TemplateSelector:
    """
    A template selector for consistent or random template selection.
    
    This class can be used to either:
    1. Use a fixed template for all samples
    2. Randomly select a template per sample
    3. Use a deterministic random template based on sample index
    """
    
    def __init__(
        self,
        use_temporal_tokens: bool = False,
        template: str = None,
        random_selection: bool = True,
        seed: int = None,
    ):
        """
        Initialize the template selector.
        
        Args:
            use_temporal_tokens: Whether to use temporal token templates.
            template: Optional fixed template to use. If None, will select randomly.
            random_selection: Whether to randomly select templates per sample.
            seed: Optional random seed for reproducibility.
        """
        self.use_temporal_tokens = use_temporal_tokens
        self.templates = get_prompt_templates(use_temporal_tokens)
        self.random_selection = random_selection
        self.seed = seed
        
        # If a fixed template is provided, use it
        if template is not None:
            self.fixed_template = template
            self.random_selection = False
        else:
            self.fixed_template = None
            
        # Initialize RNG if seed is provided
        if seed is not None:
            self.rng = random.Random(seed)
        else:
            self.rng = random
    
    def get_template(self, sample_idx: int = None) -> str:
        """
        Get a template for a sample.
        
        Args:
            sample_idx: Optional sample index for deterministic selection.
        
        Returns:
            Selected template string.
        """
        if self.fixed_template is not None:
            return self.fixed_template
        
        if not self.random_selection:
            # Use the first template as default
            return self.templates[0]
        
        # Random selection
        if sample_idx is not None and self.seed is not None:
            # Deterministic selection based on sample index
            # Note: We create a new Random instance per sample_idx to ensure
            # each sample gets a consistent template across training runs.
            # This is intentional for reproducibility - the overhead is negligible
            # (<1μs per call) and ensures true determinism even if samples are
            # accessed in different orders during distributed training.
            rng = random.Random(self.seed + sample_idx)
            return rng.choice(self.templates)
        
        # Fully random selection
        return self.rng.choice(self.templates)
    
    def format(self, query: str, sample_idx: int = None) -> str:
        """
        Get and format a template for a sample.
        
        Args:
            query: The query text to insert.
            sample_idx: Optional sample index for deterministic selection.
        
        Returns:
            Formatted prompt string.
        """
        template = self.get_template(sample_idx)
        return format_prompt(template, query)
