"""
prompted._common.service

Contains the `Service` class, which is the base class for the in-memory and
persistent service modules available in `prompted`.
"""

import logging
from pathlib import Path
from datetime import datetime
from typing import (
    Any,
    Dict,
    List,
    Generic,
    Literal,
    Optional,
    TypeVar,
    Type,
    Union
)

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pydantic import BaseModel
from sqlmodel import (
    Field,
    SQLModel,
    Session,
    select,
    create_engine
)


logger = logging.getLogger(__name__)


ServiceSchemaType = TypeVar(
    "ServiceSchemaType",
    bound=SQLModel | BaseModel | Any
)


class Service(ABC, Generic[ServiceSchemaType]):
    """
    Base class for all services that are available within the
    `prompted` package. Provides common functionality for in-memory
    and persistent (SQLModel) storage within services,
    that are partitioned by user and session ID's.
    """

    def __init__(
        self,
        location : Literal["memory", "persistent"] = "memory",
        url : Optional[Path | str] = None,
    ):
        """
        Initializes the base Service class.

        Parameters:
            location : Literal["memory", "persistent"], default="memory"
                The location of the service.
            url : Optional[Path | str], default=None
                The URL of the service.
        """
        self.location = location

        # TODO:
        # implement better split between 
        # this logic
        self._in_memory_data : Dict[str, Dict[str, Any]] = {}
        self._engine = None

        if self.location == "persistent":
            if not url:
                raise ValueError(
                    "`url` must be provided when initializing a "
                    "persistent service."
                )
            self._initialize_persistent_service(url)
            
    def _initialize_persistent_service(
        self,
        url : Path | str,
    ):
        """
        Initializes the persistent service.
        """
        try:
            self._engine = create_engine(url)

            SQLModel.metadata.create_all(
                self._engine
            )
        except Exception as e:
            raise ValueError(
                "Failed to initialize the persistent service. "
                f"Error: {e}"
            ) from e
        
    def _get_session(self) -> Session:
        """
        Provides a SQLModel session for persistent operations.
        Raises a RuntimeError if the service is not in persistent mode or engine is not initialized.
        """
        if not self._engine:
            raise RuntimeError("Database engine not initialized. Service is not in persistent mode or setup failed.")
        return Session(self._engine)
    
    @abstractmethod
    def _add_item_persistent(self, user_id: str, session_id: str, item: ServiceSchemaType):
        """
        Abstract method to add a service-specific item to persistent storage.
        To be implemented by subclasses.
        """
        pass

    @abstractmethod
    def _get_items_persistent(self, user_id: str, session_id: str) -> List[ServiceSchemaType]:
        """
        Abstract method to retrieve service-specific items from persistent storage.
        To be implemented by subclasses.
        """
        pass

    @abstractmethod
    def _clear_items_persistent(self, user_id: str, session_id: str):
        """
        Abstract method to clear service-specific items for a session from persistent storage.
        To be implemented by subclasses.
        """
        pass

    @abstractmethod
    def _clear_user_data_persistent(self, user_id: str):
        """
        Abstract method to clear all service-specific data for a user from persistent storage.
        To be implemented by subclasses.
        """
        pass
        
    # Public methods that handle mode switching and call abstract methods
    def add(self, user_id: str, session_id: str, item: ServiceSchemaType):
        """
        Adds an item to the service, associated with a user and session.
        """
        if self.location == "memory":
            if user_id not in self._in_memory_data:
                self._in_memory_data[user_id] = {}
            if session_id not in self._in_memory_data[user_id]:
                # Initialize the session-specific storage.
                # For history, this would be a list. For other services, it might be a dict.
                # The subclass would typically manage the structure within this 'Any'.
                self._in_memory_data[user_id][session_id] = [] # Default to list, subclass can override behavior
            self._in_memory_data[user_id][session_id].append(item)
            logger.debug(f"Added item for user '{user_id}', session '{session_id}' (in-memory).")
        else: # persistent
            self._add_item_persistent(user_id, session_id, item)

    def get(self, user_id: str, session_id: str) -> List[Any]:
        """
        Retrieves items for a given user and session.
        Returns an empty list if no data is found for the given user/session.
        """
        if self.location == "memory":
            return self._in_memory_data.get(user_id, {}).get(session_id, []).copy()
        else: # persistent
            return self._get_items_persistent(user_id, session_id)

    def clear(self, user_id: str, session_id: str):
        """
        Clears items for a given user and session.
        """
        if self.location == "memory":
            if user_id in self._in_memory_data and session_id in self._in_memory_data[user_id]:
                del self._in_memory_data[user_id][session_id]
                logger.debug(f"Cleared items for user '{user_id}', session '{session_id}' (in-memory).")
        else: # persistent
            self._clear_items_persistent(user_id, session_id)

    def clear_user_data(self, user_id: str):
        """
        Clears all data for a specific user across all sessions for this service.
        """
        if self.location == "memory":
            if user_id in self._in_memory_data:
                del self._in_memory_data[user_id]
                logger.debug(f"Cleared all data for user '{user_id}' (in-memory).")
        else: # persistent
            self._clear_user_data_persistent(user_id)


__all__ = [
    "Service",
    "ServiceSchemaType"
]