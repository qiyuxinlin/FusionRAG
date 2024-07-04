from asyncio import Lock
from typing import Dict, Optional

from server.schemas.assistants.runs import RunObject
from server.schemas.base import ObjectID
from server.config.log import logger

# from .job import TransformersInferenceContext
# from .transformers import TransformersInterface,get_interface

class TransformersInferenceContext:
    pass

class TransformersInterface:
    pass

def get_interface():
    pass

class ContextManager:
    lock: Lock
    threads_context: Dict[ObjectID, TransformersInferenceContext]

    def __init__(self) -> None:
        logger.debug(f"Creating Context Manager")
        self.lock = Lock()
        self.threads_context = {}

        pass

    async def get_context_by_run_object(self, run: RunObject) -> TransformersInferenceContext:
        async with self.lock:
            logger.debug(f"keys {self.threads_context.keys()}")
            if run.thread_id not in self.threads_context:
                logger.debug(f"new inference context {run.thread_id}")
                new_context = TransformersInferenceContext(run)
                self.threads_context[run.thread_id] = new_context
                # self.threads_context[run.thread_id] = ExllamaInferenceContext(run)
            re = self.threads_context[run.thread_id]
            re.update_by_run(run)
            return re

    async def get_context_by_thread_id(self, thread_id: ObjectID) -> Optional[TransformersInferenceContext]:
        async with self.lock:
            if thread_id in self.threads_context:
                logger.debug(f'found context for thread {thread_id}')
                return self.threads_context[thread_id]
            else:
                logger.debug(f'no context for thread {thread_id}')
                return None
            
            
    


context_manager: ContextManager = ContextManager()


def get_thread_context_manager() -> ContextManager:
    return context_manager
