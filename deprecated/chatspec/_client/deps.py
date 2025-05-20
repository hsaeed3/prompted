"""
💭 chatspec._client.deps
"""

from typing import Any, Dict, List, Optional

from ..params import InstructorModeParam

__all__ = [
    "get_client_deps",
    "ClientDeps",
]


class ClientDeps:
    """
    A class that contains the dependencies for the `Client`
    class.
    """

    is_litellm_initialized = False
    """Whether `litellm` has been initialized."""
    completion = None
    """`litellm.completion`"""
    completion_async = None
    """`litellm.acompletion`"""
    batch_completion = None
    """`litellm.batch_completion`"""
    batch_completion_models = None
    """`litellm.batch_completion_models_all_responses`"""
    embedding = None
    """`litellm.embedding`"""
    image_generation = None
    """`litellm.image_generation`"""
    audio_generation = None
    """`litellm.speech`"""
    audio_transcription = None
    """`litellm.transcription`"""

    is_instructor_initialized = False
    """Whether `instructor` has been initialized."""
    instructor_sync = None
    """Synchronous instructor client."""
    instructor_async = None
    """Asynchronous instructor client."""
    instructor_mode = None
    """Reference to `instructor.Mode`."""

    def __init__(self):
        pass

    def initialize_litellm(self):
        """
        Initialize the litellm client.
        """
        try:
            import litellm
        except ImportError:
            raise ImportError(
                "`litellm` is not installed. Please install it with `pip install litellm`."
                f"Or use `pip install chatspec[client]` to install it."
            )

        litellm.drop_params = True
        litellm.modify_params = True
        litellm.ssl_verify = False

        self.completion = staticmethod(litellm.completion)
        self.completion_async = staticmethod(litellm.acompletion)
        self.batch_completion = staticmethod(litellm.batch_completion)
        self.batch_completion_models = staticmethod(
            litellm.batch_completion_models_all_responses
        )
        self.embedding = staticmethod(litellm.embedding)
        self.image_generation = staticmethod(litellm.image_generation)
        self.audio_generation = staticmethod(litellm.speech)
        self.audio_transcription = staticmethod(litellm.transcription)
        self.is_litellm_initialized = True

    def initialize_instructor(self):
        """
        Initialize the instructor client.
        """
        try:
            import instructor
        except ImportError:
            raise ImportError(
                "`instructor` is not installed. Please install it with `pip install instructor`."
                f"Or use `pip install chatspec[client]` to install it."
            )

        if not self.is_litellm_initialized:
            self.initialize_litellm()

        self.instructor_sync = instructor.from_litellm(self.completion)
        self.instructor_async = instructor.from_litellm(
            self.completion_async
        )
        self.instructor_mode = instructor.Mode

        self.is_instructor_initialized = True

    @property
    def instructor_mode(self) -> str:
        """
        Get the instructor mode.
        """
        if not self.is_instructor_initialized:
            self.initialize_instructor()

        return self.instructor_sync.mode.value

    @instructor_mode.setter
    def instructor_mode(self, mode: InstructorModeParam = "tool_call"):
        """
        Set the instructor mode.
        """
        if not self.is_instructor_initialized:
            self.initialize_instructor()

        self.instructor_sync.mode = self.instructor_mode(mode)

    @property
    def instructor_async_mode(self) -> str:
        """
        Get the instructor async mode.
        """
        if not self.is_instructor_initialized:
            self.initialize_instructor()

        return self.instructor_async.mode.value

    @instructor_async_mode.setter
    def instructor_async_mode(
        self, mode: InstructorModeParam = "tool_call"
    ):
        """
        Set the instructor async mode.
        """
        if not self.is_instructor_initialized:
            self.initialize_instructor()

        self.instructor_async.mode = self.instructor_mode(mode)


CLIENT_DEPS: ClientDeps = ClientDeps()
"""Singleton instance of `ClientDeps`."""


def get_client_deps() -> ClientDeps:
    """
    Retrieves the singleton instance of `ClientDeps`.
    """
    if not CLIENT_DEPS.is_instructor_initialized:
        CLIENT_DEPS.initialize_instructor()

    return CLIENT_DEPS
