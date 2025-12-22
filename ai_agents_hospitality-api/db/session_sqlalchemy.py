import contextlib
from functools import wraps
from typing import AsyncIterator, Callable, Any

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    AsyncEngine,
    create_async_engine,
    async_sessionmaker,
    AsyncConnection,
)

from util.logger_config import logger


# ============================================================================
# DatabaseSessionManager - Main session management class
# ============================================================================

class DatabaseSessionManager:
    def __init__(self, host: str):
        self.engine: AsyncEngine | None = create_async_engine(host, echo=True)
        self._sessionmaker: async_sessionmaker[AsyncSession] = async_sessionmaker(
            autocommit=False, autoflush=False, bind=self.engine
        )

    async def close(self):
        """
        Closes the database engine and releases associated resources. Should be called during service shutdown or when
        resetting the connection.
        """
        if self.engine is None:
            raise Exception(msg = "Database engine is not available. Ensure it has "
                                                                     "been properly initialized before attempting to close it.")
        await self.engine.dispose()
        self.engine = None
        self._sessionmaker = None  # type: ignore

    @contextlib.asynccontextmanager
    async def connect(self) -> AsyncIterator[AsyncConnection]:
        """
        Creates a low-level direct connection to the database engine. Used to execute raw SQL statements without using
        the ORM. Manages the transactional context (begin/rollback) manually.
        """
        if self.engine is None:
            raise e
        async with self.engine.begin() as connection:
            try:
                yield connection
            except SQLAlchemyError as e:
                await connection.rollback()
                logger.error("Connection error occurred")
                raise e

    @contextlib.asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        """
        Creates a high-level ORM session bound to the engine. Allows working with mapped models (entities) and automatically
        manages transactions.Recommended method for regular read/write operations in the system.
        """
        if not self._sessionmaker:
            logger.error("Sessionmaker is not available")
            raise Exception("Sessionmaker is not available. Ensure the database engine has been properly initialized before creating sessions.")

        session = self._sessionmaker()
        try:
            yield session
        except SQLAlchemyError as e:
            await session.rollback()
            logger.error(f"Session error could not be established {e}")
            raise e
        finally:
            await session.close()


# ============================================================================
# Context Managers - Session helpers with transaction control
# ============================================================================

@contextlib.asynccontextmanager
async def get_db_session(sessionmanager: DatabaseSessionManager) -> AsyncIterator[AsyncSession]:
    """
    Context for database operations without automatic commit.
        - Performs a rollback if an error occurs.
        - Ensures the session is always closed.
    """
    async with sessionmanager.session() as session:
        try:
            yield session
        except Exception as e:
            logger.error(f"Error occurred on db transaction: {e}")
            await session.rollback()
            raise e
        finally:
            await session.close()


@contextlib.asynccontextmanager
async def transaction_db_async(sessionmanager: DatabaseSessionManager) -> AsyncIterator[AsyncSession]:
    """
    Behavior:
    - Commits at the end if no exceptions occur.
    - Automatically rolls back if an exception is raised.
    - Ensures the session is always closed at the end.
    """
    async with sessionmanager.session() as session:
        try:
            yield session
            await session.commit()
        except Exception as e:
            logger.error(f"Error occurred on db transaction: {e}")
            await session.rollback()
            raise e
        finally:
            await session.close()


# ============================================================================
# Decorators - Automatic session injection for class methods
# ============================================================================

def transactional(func: Callable) -> Callable:
    """
    Decorator that wraps an async function with a database transaction.
    
    Usage:
        @transactional
        async def create_user(self, db_session: AsyncSession, name: str, email: str) -> User:
            new_user = User(name=name, email=email)
            db_session.add(new_user)
            return new_user
    
    The decorator:
    - Injects 'db_session: AsyncSession' as the first parameter after 'self'
    - Automatically commits on success
    - Automatically rolls back on exception
    - Always closes the session
    - Uses self.db_manager from the injected DatabaseSessionManager
    """
    @wraps(func)
    async def wrapper(self, *args, **kwargs) -> Any:
        logger.debug(f"Starting transaction: {func.__name__}")
        async with transaction_db_async(self.db_manager) as db_session:
            result = await func(self, db_session, *args, **kwargs)
            logger.debug(f"Transaction committed: {func.__name__}")
            return result

    
    return wrapper


def with_session(func: Callable) -> Callable:
    """
    Decorator that provides a database session without automatic commit.
    Useful for read-only operations or operations where the caller manages transactions.
    
    Usage:
        @with_session
        async def get_user(self, db_session: AsyncSession, user_id: str) -> User:
            result = await db_session.execute(select(User).where(User.id == user_id))
            return result.scalar_one_or_none()
    
    The decorator:
    - Injects 'db_session: AsyncSession' as the first parameter after 'self'
    - Rolls back on exception
    - Always closes the session
    - Does NOT auto-commit (read-only operations) or lets caller manage transactions
    - Uses self.db_manager from the injected DatabaseSessionManager
    """
    @wraps(func)
    async def wrapper(self, *args, **kwargs) -> Any:
        logger.debug(f"Starting transaction with no auto commit: {func.__name__}")
        async with get_db_session(self.db_manager) as db_session:
            result = await func(self, db_session, *args, **kwargs)
            logger.debug(f"Transaction with no autocommit ended: {func.__name__}")
            return result
    
    return wrapper