from asyncio import Lock
from typing import Dict, Optional

from server.backend.base import ThreadContext
from server.schemas.assistants.runs import RunObject
from server.schemas.base import ObjectID
from server.config.log import logger
from config.config import Config

conf = Config()

logger.warn(f'Backend Type {conf.backend_type}')

if conf.backend_type=='transformers':
    from .interfaces.transformers import TransformersThreadContext as TContext, TransformersInterface as BackendInterface
elif conf.backend_type == 'exllamav2':
    from .interfaces.exllamav2 import ExllamaThreadContext as TContext, ExllamaInterface as BackendInterface
else:
    raise NotImplementedError(f'{conf.backend_type} not implemented')

class globalInterface:
    interface:BackendInterface   
def get_interface()->BackendInterface:
    return globalInterface.interface

class ThreadContextManager:
    lock: Lock
    threads_context: Dict[ObjectID, ThreadContext]

    def __init__(self) -> None:
        logger.debug(f"Creating Context Manager")
        self.lock = Lock()
        self.threads_context = {}

        pass

    async def get_context_by_run_object(self, run: RunObject) -> ThreadContext:
        async with self.lock:
            logger.debug(f"keys {self.threads_context.keys()}")
            if run.thread_id not in self.threads_context:
                logger.debug(f"new inference context {run.thread_id}")
                new_context = TContext(run,get_interface())
                self.threads_context[run.thread_id] = new_context
                # self.threads_context[run.thread_id] = ExllamaInferenceContext(run)
            re = self.threads_context[run.thread_id]
            re.update_by_run(run)
            return re

    async def get_context_by_thread_id(self, thread_id: ObjectID) -> Optional[ThreadContext]:
        async with self.lock:
            if thread_id in self.threads_context:
                logger.debug(f'found context for thread {thread_id}')
                return self.threads_context[thread_id]
            else:
                logger.debug(f'no context for thread {thread_id}')
                return None
            
            
context_manager: ThreadContextManager = ThreadContextManager()


def get_thread_context_manager() -> ThreadContextManager:
    return context_manager
