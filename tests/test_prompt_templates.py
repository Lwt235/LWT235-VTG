"""
Tests for prompt template module.
"""

import pytest

from vtg_datasets.prompt_templates import (
    get_prompt_templates,
    get_random_template,
    format_prompt,
    TemplateSelector,
    STANDARD_TEMPLATES,
    TEMPORAL_TOKEN_TEMPLATES,
)


class TestPromptTemplates:
    """Tests for prompt template functions."""

    def test_get_prompt_templates_standard(self):
        """Test getting standard templates."""
        templates = get_prompt_templates(use_temporal_tokens=False)
        
        assert isinstance(templates, list)
        assert len(templates) > 0
        assert len(templates) == len(STANDARD_TEMPLATES)
        assert all(isinstance(t, str) for t in templates)
        assert all("{query}" in t for t in templates)

    def test_get_prompt_templates_temporal(self):
        """Test getting temporal token templates."""
        templates = get_prompt_templates(use_temporal_tokens=True)
        
        assert isinstance(templates, list)
        assert len(templates) > 0
        assert len(templates) == len(TEMPORAL_TOKEN_TEMPLATES)
        assert all(isinstance(t, str) for t in templates)
        assert all("{query}" in t for t in templates)

    def test_get_random_template(self):
        """Test getting random template."""
        template = get_random_template(use_temporal_tokens=False)
        
        assert isinstance(template, str)
        assert "{query}" in template
        assert template in STANDARD_TEMPLATES

    def test_get_random_template_with_seed(self):
        """Test deterministic template selection with seed."""
        template1 = get_random_template(use_temporal_tokens=False, seed=42)
        template2 = get_random_template(use_temporal_tokens=False, seed=42)
        
        assert template1 == template2

    def test_format_prompt(self):
        """Test formatting a template with query."""
        template = "Find when \"{query}\" occurs in the video."
        query = "a person walks"
        
        result = format_prompt(template, query)
        
        assert "a person walks" in result
        assert "{query}" not in result

    def test_template_variety(self):
        """Test that we have multiple diverse templates."""
        standard_templates = get_prompt_templates(use_temporal_tokens=False)
        temporal_templates = get_prompt_templates(use_temporal_tokens=True)
        
        # Should have at least 5 templates for variety
        assert len(standard_templates) >= 5
        assert len(temporal_templates) >= 5
        
        # Templates should be different from each other
        assert len(set(standard_templates)) == len(standard_templates)
        assert len(set(temporal_templates)) == len(temporal_templates)


class TestTemplateSelector:
    """Tests for TemplateSelector class."""

    def test_fixed_template(self):
        """Test using a fixed template."""
        fixed_template = "Custom template: {query}"
        selector = TemplateSelector(
            use_temporal_tokens=False,
            template=fixed_template,
        )
        
        # Should always return the fixed template
        assert selector.get_template(0) == fixed_template
        assert selector.get_template(100) == fixed_template

    def test_random_selection(self):
        """Test random template selection."""
        selector = TemplateSelector(
            use_temporal_tokens=False,
            random_selection=True,
            seed=42,
        )
        
        # Get templates for different samples
        template1 = selector.get_template(0)
        template2 = selector.get_template(1)
        
        # Should be valid templates
        assert template1 in STANDARD_TEMPLATES
        assert template2 in STANDARD_TEMPLATES

    def test_deterministic_selection(self):
        """Test deterministic template selection with seed."""
        selector1 = TemplateSelector(
            use_temporal_tokens=False,
            random_selection=True,
            seed=42,
        )
        selector2 = TemplateSelector(
            use_temporal_tokens=False,
            random_selection=True,
            seed=42,
        )
        
        # Same seed should give same templates for same indices
        for idx in range(10):
            assert selector1.get_template(idx) == selector2.get_template(idx)

    def test_format_method(self):
        """Test the format convenience method."""
        selector = TemplateSelector(
            use_temporal_tokens=False,
            template="Test template: {query}",
        )
        
        result = selector.format("my query", sample_idx=0)
        
        assert "my query" in result
        assert "{query}" not in result

    def test_no_random_selection(self):
        """Test with random selection disabled."""
        selector = TemplateSelector(
            use_temporal_tokens=False,
            random_selection=False,
        )
        
        # Should use first template consistently
        template1 = selector.get_template(0)
        template2 = selector.get_template(100)
        
        assert template1 == template2
        assert template1 == STANDARD_TEMPLATES[0]

    def test_temporal_token_templates(self):
        """Test selector with temporal token templates."""
        selector = TemplateSelector(
            use_temporal_tokens=True,
            random_selection=True,
            seed=42,
        )
        
        template = selector.get_template(0)
        
        assert template in TEMPORAL_TOKEN_TEMPLATES
        assert "temporal token" in template.lower() or "temporal" in template.lower()
