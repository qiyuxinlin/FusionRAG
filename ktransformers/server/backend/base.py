from asyncio import Queue
from enum import Enum
import sys, os
from typing import AsyncIterator, Dict, List, Optional, Tuple

import torch

from server.config.log import logger
from server.crud.assistants.assistants import AssistantDatabaseManager
from server.crud.assistants.messages import MessageDatabaseManager
from server.crud.assistants.runs import RunsDatabaseManager
from server.crud.assistants.threads import ThreadsDatabaseManager
from server.exceptions import request_error
from server.schemas.assistants.assistants import AssistantObject
from server.schemas.assistants.messages import MessageCreate, MessageObject, Role
from server.schemas.assistants.runs import RunObject
from server.schemas.assistants.threads import ThreadObject
from server.schemas.base import ObjectID, Order
from server.utils.multi_timer import Profiler


from .args import ConfigArgs,default_args



class BackendInterfaceBase:
    args: ConfigArgs


    # profile
    profiler:Profiler = Profiler()

    def __init__(self, args:ConfigArgs = default_args):
        raise NotImplementedError

    def tokenize_prompt(self,prompt:str)->torch.Tensor:
        raise NotImplementedError

    
    async def work(self,local_messages,request_unique_id:Optional[str])->AsyncIterator:
        raise NotImplementedError


    def report_last_time_performance(self):
        tokenize_time = self.profiler.get_timer_sec('tokenize')
        prefill_time = self.profiler.get_timer_sec('prefill')
        decode_time = self.profiler.get_timer_sec('decode')
        prefill_count = self.profiler.get_counter('prefill')
        decode_count = self.profiler.get_counter('decode')

        logger.info(f'Performance(T/s): prefill {prefill_count/prefill_time}, decode {decode_count/decode_time}. Time(s): tokenize {tokenize_time}, prefill {prefill_time}, decode {decode_time}')



class ThreadContext:

    args: ConfigArgs
    # Assistant Logic
    assistant: Optional[AssistantObject] = None
    related_threads : List[ThreadObject]
    thread: ThreadObject
    messages: List[MessageObject] = [] 
    run: RunObject

    interface: Optional[BackendInterfaceBase] = None
     
    queue: Optional[Queue] = None
    timer: Profiler = Profiler()

    def __init__(self, run: RunObject, args: ConfigArgs = default_args) -> None:
        self.args = args
        self.thread_manager = ThreadsDatabaseManager()
        self.message_manager = MessageDatabaseManager()
        self.runs_manager = RunsDatabaseManager()
        self.assistant_manager = AssistantDatabaseManager()
        self.thread = self.thread_manager.db_get_thread_by_id(run.thread_id)
        self.assistant = self.assistant_manager.db_get_assistant_by_id(run.assistant_id)
        self.messages = self.message_manager.db_list_messages_of_thread(run.thread_id,order=Order.ASC)
        logger.debug(f"{len(self.messages)} messages loaded from databsae")
        self.interface = self.get_interface()
        self.update_by_run(run,args)

    def update_by_run(self,run:RunObject,args:ConfigArgs = default_args):
        self.run = run 
        self.args = args
       
    def put_user_message(self, message: MessageObject):
        assert (
            message.role.is_user() and message.thread_id == self.thread.id and message.status == MessageObject.Status.in_progress
        )
        self.messages.append(message)

    def delete_user_message(self,message_id: ObjectID):
        self.messages = [m for m in self.messages if m.id != message_id]


    def get_interface(self):
        raise NotImplementedError

    async def get_local_messages(self):
        raise NotImplementedError


    async def work(self)->AsyncIterator:
        logger.debug('start working')
        user_message = self.messages[-1]
        if not user_message.role.is_user():
            raise request_error('user must talk before LLM can talk')
        user_message.status = MessageObject.Status.completed
        user_message.sync_db()

        local_messages = self.get_local_messages() # must get this before we interseted reply_message


        response_str_count = 0  
        reply_message = self.message_manager.create_message_object(
                            self.thread.id,
                            self.run.id,
                            MessageCreate(role=Role.assistant, content=""),    
                        )
        reply_message.assistant_id = self.assistant.id
        self.messages.append(reply_message) 

        yield reply_message.stream_response_with_event(MessageObject.Status.created)
        yield reply_message.stream_response_with_event(MessageObject.Status.in_progress)
        yield self.run.stream_response_with_event(RunObject.Status.in_progress)

        async for token in self.interface.work(local_messages,self.thread.id):     
            if self.run.status == RunObject.Status.cancelling:
                logger.warn(f'Run {self.run.id} cancelling')
                break
            yield reply_message.append_message_delta(token)
            response_str_count+=1
        
        if self.run.status == RunObject.Status.cancelling:
            yield self.run.stream_response_with_event(RunObject.Status.cancelled)
            yield reply_message.stream_response_with_event(MessageObject.Status.incomplete)
        elif self.run.status == RunObject.Status.in_progress:
            yield self.run.stream_response_with_event(RunObject.Status.completed)
            yield reply_message.stream_response_with_event(MessageObject.Status.completed)
        else:
            raise NotImplementedError(f'{self.run.status} should not appear here')

        reply_message.sync_db()
        self.run.sync_db()