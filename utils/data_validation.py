"""
Data validation utilities for Video Temporal Localization.

Provides validation for annotation files and video data consistency.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from .logging_utils import get_logger

logger = get_logger(__name__)

# Data format field constants - used by both validation and dataset classes
REQUIRED_FIELDS = ["video", "duration", "timestamp", "sentence"]
# Note: Additional fields like qid, difficulty, pred are allowed as direct metadata fields
OPTIONAL_KNOWN_FIELDS = ["video_start", "video_end", "qid", "difficulty", "pred"]


class DataValidationError(Exception):
    """Exception raised for data validation errors."""
    pass


class DataValidator:
    """
    Validator for video temporal grounding dataset.
    
    Validates annotation files and checks data consistency.
    """
    
    REQUIRED_FIELDS = REQUIRED_FIELDS
    OPTIONAL_FIELDS = OPTIONAL_KNOWN_FIELDS
    
    def __init__(
        self,
        video_dir: Optional[str] = None,
        check_video_exists: bool = True,
        strict_mode: bool = False,
    ):
        """
        Initialize the validator.
        
        Args:
            video_dir: Base directory for video files.
            check_video_exists: Whether to verify video files exist.
            strict_mode: If True, raise errors; if False, log warnings.
        """
        self.video_dir = Path(video_dir) if video_dir else None
        self.check_video_exists = check_video_exists
        self.strict_mode = strict_mode
        self.errors: List[str] = []
        self.warnings: List[str] = []
    
    def validate_sample(
        self,
        sample: Dict[str, Any],
        line_num: Optional[int] = None,
    ) -> bool:
        """
        Validate a single annotation sample.
        
        Args:
            sample: Dictionary containing annotation data.
            line_num: Line number in the source file (for error messages).
        
        Returns:
            True if valid, False otherwise.
        """
        prefix = f"Line {line_num}: " if line_num is not None else ""
        is_valid = True
        
        # Check required fields
        for field in self.REQUIRED_FIELDS:
            if field not in sample:
                self._add_error(f"{prefix}Missing required field: {field}")
                is_valid = False
        
        if not is_valid:
            return False
        
        # Validate video field
        video_path = sample.get("video", "")
        if not isinstance(video_path, str) or not video_path:
            self._add_error(f"{prefix}Invalid video path: {video_path}")
            is_valid = False
        elif self.check_video_exists and self.video_dir:
            full_path = self.video_dir / video_path
            if not full_path.exists():
                self._add_warning(f"{prefix}Video file not found: {full_path}")
        
        # Validate duration
        duration = sample.get("duration")
        if not isinstance(duration, (int, float)) or duration <= 0:
            self._add_error(f"{prefix}Invalid duration: {duration}")
            is_valid = False
        
        # Validate timestamp
        timestamp = sample.get("timestamp")
        if not self._validate_timestamp(timestamp, duration, prefix):
            is_valid = False
        
        # Validate sentence
        sentence = sample.get("sentence", "")
        if not isinstance(sentence, str) or not sentence.strip():
            self._add_error(f"{prefix}Invalid or empty sentence")
            is_valid = False
        
        # Validate optional fields
        video_start = sample.get("video_start")
        if video_start is not None:
            if not isinstance(video_start, (int, float)) or video_start < 0:
                self._add_error(f"{prefix}Invalid video_start: {video_start}")
                is_valid = False
        
        video_end = sample.get("video_end")
        if video_end is not None:
            if not isinstance(video_end, (int, float)) or video_end <= 0:
                self._add_error(f"{prefix}Invalid video_end: {video_end}")
                is_valid = False
        
        if video_start is not None and video_end is not None:
            if video_start >= video_end:
                self._add_error(f"{prefix}video_start >= video_end")
                is_valid = False
        
        return is_valid
    
    def _validate_timestamp(
        self,
        timestamp: Any,
        duration: float,
        prefix: str = "",
    ) -> bool:
        """Validate timestamp field."""
        if not isinstance(timestamp, (list, tuple)) or len(timestamp) != 2:
            self._add_error(f"{prefix}Timestamp must be a list of two floats")
            return False
        
        start, end = timestamp
        
        if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
            self._add_error(f"{prefix}Timestamp values must be numeric")
            return False
        
        if start < 0:
            self._add_error(f"{prefix}Timestamp start cannot be negative: {start}")
            return False
        
        if end <= start:
            self._add_error(f"{prefix}Timestamp end must be greater than start")
            return False
        
        if end > duration:
            self._add_warning(f"{prefix}Timestamp end ({end}) exceeds duration ({duration})")
        
        return True
    
    def _add_error(self, message: str):
        """Add an error message."""
        self.errors.append(message)
        if self.strict_mode:
            raise DataValidationError(message)
        logger.error(message)
    
    def _add_warning(self, message: str):
        """Add a warning message."""
        self.warnings.append(message)
        logger.warning(message)
    
    def validate_file(
        self,
        annotation_file: Union[str, Path],
        max_errors: int = 100,
    ) -> Tuple[int, int, int]:
        """
        Validate an entire annotation file.
        
        Args:
            annotation_file: Path to JSONL annotation file.
            max_errors: Maximum number of errors before stopping.
        
        Returns:
            Tuple of (total_samples, valid_samples, error_count).
        """
        annotation_file = Path(annotation_file)
        
        if not annotation_file.exists():
            raise FileNotFoundError(f"Annotation file not found: {annotation_file}")
        
        self.errors = []
        self.warnings = []
        
        total_samples = 0
        valid_samples = 0
        
        logger.info(f"Validating annotation file: {annotation_file}")
        
        with open(annotation_file, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                if len(self.errors) >= max_errors:
                    logger.warning(f"Reached maximum error limit ({max_errors}), stopping validation")
                    break
                
                line = line.strip()
                if not line:
                    continue
                
                try:
                    sample = json.loads(line)
                except json.JSONDecodeError as e:
                    self._add_error(f"Line {line_num}: Invalid JSON: {e}")
                    total_samples += 1
                    continue
                
                total_samples += 1
                if self.validate_sample(sample, line_num):
                    valid_samples += 1
        
        logger.info(
            f"Validation complete: {valid_samples}/{total_samples} valid samples, "
            f"{len(self.errors)} errors, {len(self.warnings)} warnings"
        )
        
        return total_samples, valid_samples, len(self.errors)
    
    def get_report(self) -> Dict[str, Any]:
        """Get validation report."""
        return {
            "errors": self.errors,
            "warnings": self.warnings,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
        }


def validate_annotation_file(
    annotation_file: Union[str, Path],
    video_dir: Optional[str] = None,
    check_video_exists: bool = False,
    strict_mode: bool = False,
) -> Tuple[int, int, int]:
    """
    Convenience function to validate an annotation file.
    
    Args:
        annotation_file: Path to JSONL annotation file.
        video_dir: Base directory for video files.
        check_video_exists: Whether to verify video files exist.
        strict_mode: If True, raise errors on first invalid sample.
    
    Returns:
        Tuple of (total_samples, valid_samples, error_count).
    """
    validator = DataValidator(
        video_dir=video_dir,
        check_video_exists=check_video_exists,
        strict_mode=strict_mode,
    )
    return validator.validate_file(annotation_file)
