
import enum
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


class ErrorSeverity(enum.Enum):
    """Enum for error severity levels."""
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class ApplicationError(Exception):
    """Custom error class for application-specific errors."""
    message: str
    details: str = ""
    severity: ErrorSeverity = ErrorSeverity.ERROR
    recovery_hint: Optional[str] = None
    original_error: Optional[Exception] = None
    source: Optional[str] = None

    def __str__(self):
        """String representation of the error."""
        error_str = self.message
        if self.details:
            error_str += f": {self.details}"
        if self.recovery_hint:
            error_str += f" ({self.recovery_hint})"
        return error_str


# class ProcessingError(ApplicationError):
#     """Specific error class for processing-related errors."""
#
#     def __init__(
#             self,
#             message: str,
#             details: str = "",
#             severity: ErrorSeverity = ErrorSeverity.ERROR,
#             recovery_hint: Optional[str] = None
#     ):
#         super().__init__(
#             message=message,
#             details=details,
#             severity=severity,
#             recovery_hint=recovery_hint
#         )


class ErrorHandlingMixin:
    """Mixin class providing error handling functionality."""

    def __init__(self):
        self._error_handlers = []


    def handle_error(self, error: ApplicationError):
        """Handle an error by passing it to all registered handlers."""
        error_str = str(error)

        # Log the error
        if error.severity == ErrorSeverity.CRITICAL:
            logger.critical(error_str, exc_info=error.original_error)
        elif error.severity == ErrorSeverity.ERROR:
            logger.error(error_str, exc_info=error.original_error)
        elif error.severity == ErrorSeverity.WARNING:
            logger.warning(error_str)
        else:
            logger.info(error_str)

        # Notify handlers
        for handler in self._error_handlers:
            try:
                handler(error)
            except Exception as e:
                logger.error(f"Error in error handler: {e}")

    @staticmethod
    def create_error(
            message: str,
            details: str = "",
            severity: ErrorSeverity = ErrorSeverity.ERROR,
            recovery_hint: Optional[str] = None,
            original_error: Optional[Exception] = None,
            source: Optional[str] = None
    ) -> ApplicationError:
        """Create an ApplicationError with the given parameters."""
        return ApplicationError(
            message=message,
            details=details,
            severity=severity,
            recovery_hint=recovery_hint,
            original_error=original_error,
            source=source
        )