#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   exceptions.py
@Time    :   2025/08/01 11:15:43
@Author  :   Dawn Li
@Version :   1.0
@Contact :   dawnli619215645@gmail.com
@Desc    :   Custom exception classes for flow module including FlowBaseException,
             CoordinateParsingError, DirectionTokenError, and TokenPreprocessingError
             with context information
"""

from typing import Any, Dict, Optional


class FlowBaseException(Exception):
    """Base exception for all flow module errors"""

    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.context = context or {}
        self.message = message

    def __str__(self) -> str:
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            return f"{self.message} (Context: {context_str})"
        return self.message


class CoordinateParsingError(FlowBaseException):
    """Raised when coordinate string parsing fails"""

    def __init__(self, coord_str: str, reason: Optional[str] = None):
        message = f"Failed to parse coordinate: {coord_str}"
        if reason:
            message += f" ({reason})"
        super().__init__(message, {"coordinate": coord_str, "reason": reason})


class DirectionTokenError(FlowBaseException):
    """Raised when direction token processing fails"""

    def __init__(self, token: str, reason: Optional[str] = None):
        message = f"Invalid direction token: {token}"
        if reason:
            message += f" ({reason})"
        super().__init__(message, {"token": token, "reason": reason})


class TokenPreprocessingError(FlowBaseException):
    """Raised when token concatenation/segmentation fails"""

    def __init__(self, tokens: Any, operation: str, reason: Optional[str] = None):
        message = f"Failed to {operation} tokens"
        if reason:
            message += f": {reason}"
        super().__init__(
            message,
            {
                "tokens": str(tokens)[:200] + "..."
                if len(str(tokens)) > 200
                else str(tokens),
                "operation": operation,
                "reason": reason,
            },
        )
