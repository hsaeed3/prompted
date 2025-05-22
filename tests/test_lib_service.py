# tests.test_lib_service

# tests the `Service` base class used for all service 
# module implementations found within the `prompted`
# package.

import pytest


def test_internal_lib_service_in_memory():
    """
    Tests the `Service` class by implementing an 'in-memory' service with
    an example schema.
    """
    from prompted._lib.service import Service
    from pydantic import BaseModel
    from typing import List, Optional
    
    class TestItem(BaseModel):
        name: str
        value: int
        description: Optional[str] = None
    
    class TestService(Service[TestItem]):
        def _add_item_persistent(self, user_id, session_id, item):
            pass  # Not needed for in-memory test
            
        def _get_items_persistent(self, user_id, session_id):
            pass  # Not needed for in-memory test
            
        def _clear_items_persistent(self, user_id, session_id):
            pass  # Not needed for in-memory test
            
        def _clear_user_data_persistent(self, user_id):
            pass  # Not needed for in-memory test
    
    # Create service instance
    service = TestService(location="memory")
    
    # Test adding items
    user_id = "test_user"
    session_id = "test_session"
    item1 = TestItem(name="item1", value=10)
    item2 = TestItem(name="item2", value=20, description="Test item")
    
    service.add(user_id, session_id, item1)
    service.add(user_id, session_id, item2)
    
    # Test retrieving items
    items = service.get(user_id, session_id)
    assert len(items) == 2
    assert items[0].name == "item1"
    assert items[1].value == 20
    
    # Test retrieving for non-existent user/session
    assert len(service.get("nonexistent", "nonexistent")) == 0
    
    # Test clearing session data
    service.clear(user_id, session_id)
    assert len(service.get(user_id, session_id)) == 0
    
    # Test adding to multiple sessions
    session_id2 = "test_session2"
    service.add(user_id, session_id, item1)
    service.add(user_id, session_id2, item2)
    
    assert len(service.get(user_id, session_id)) == 1
    assert len(service.get(user_id, session_id2)) == 1
    
    # Test clearing all user data
    service.clear_user_data(user_id)
    assert len(service.get(user_id, session_id)) == 0
    assert len(service.get(user_id, session_id2)) == 0


def test_internal_lib_service_persistent():
    """
    Tests the `Service` class by implementing a 'persistent' service with
    an example schema.
    """
    import tempfile
    import os
    from prompted._lib.service import Service
    from sqlmodel import SQLModel, Field, select
    from typing import Optional
    
    class TestItemModel(SQLModel, table=True):
        id: Optional[int] = Field(default=None, primary_key=True)
        user_id: str
        session_id: str
        name: str
        value: int
        description: Optional[str] = None
    
    class TestPersistentService(Service[TestItemModel]):
        def _add_item_persistent(self, user_id, session_id, item):
            item.user_id = user_id
            item.session_id = session_id
            with self._get_session() as session:
                session.add(item)
                session.commit()
                session.refresh(item)
            
        def _get_items_persistent(self, user_id, session_id):
            with self._get_session() as session:
                statement = select(TestItemModel).where(
                    TestItemModel.user_id == user_id,
                    TestItemModel.session_id == session_id
                )
                return session.exec(statement).all()
            
        def _clear_items_persistent(self, user_id, session_id):
            with self._get_session() as session:
                statement = select(TestItemModel).where(
                    TestItemModel.user_id == user_id,
                    TestItemModel.session_id == session_id
                )
                items = session.exec(statement).all()
                for item in items:
                    session.delete(item)
                session.commit()
            
        def _clear_user_data_persistent(self, user_id):
            with self._get_session() as session:
                statement = select(TestItemModel).where(
                    TestItemModel.user_id == user_id
                )
                items = session.exec(statement).all()
                for item in items:
                    session.delete(item)
                session.commit()
    
    # Create a temporary SQLite database
    db_file = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
    db_file.close()
    db_url = f"sqlite:///{db_file.name}"
    
    try:
        # Create service instance
        service = TestPersistentService(location="persistent", url=db_url)
        
        # Test adding items
        user_id = "test_user"
        session_id = "test_session"
        item1 = TestItemModel(name="item1", value=10, user_id="", session_id="")
        item2 = TestItemModel(name="item2", value=20, description="Test item", user_id="", session_id="")
        
        service.add(user_id, session_id, item1)
        service.add(user_id, session_id, item2)
        
        # Test retrieving items
        items = service.get(user_id, session_id)
        assert len(items) == 2
        assert items[0].name == "item1"
        assert items[1].value == 20
        
        # Test retrieving for non-existent user/session
        assert len(service.get("nonexistent", "nonexistent")) == 0
        
        # Test clearing session data
        service.clear(user_id, session_id)
        assert len(service.get(user_id, session_id)) == 0
        
        # Test adding to multiple sessions
        session_id2 = "test_session2"
        item3 = TestItemModel(name="item3", value=30, user_id="", session_id="")
        item4 = TestItemModel(name="item4", value=40, user_id="", session_id="")
        
        service.add(user_id, session_id, item3)
        service.add(user_id, session_id2, item4)
        
        assert len(service.get(user_id, session_id)) == 1
        assert len(service.get(user_id, session_id2)) == 1
        
        # Test clearing all user data
        service.clear_user_data(user_id)
        assert len(service.get(user_id, session_id)) == 0
        assert len(service.get(user_id, session_id2)) == 0
    
    finally:
        # Clean up the temporary database file
        os.unlink(db_file.name)