import sys, os
from typing import Dict, Tuple
import torch.nn as nn

from ..base import ThreadContext

sys.path.append(
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "exllamav2",
    )
)

import torch
from asyncio import Queue

from server.utils.multi_timer import Profiler
from enum import Enum

from server.crud.assistants.messages import MessageDatabaseManager
from server.crud.assistants.assistants import AssistantDatabaseManager
from server.crud.assistants.runs import RunsDatabaseManager
from server.crud.assistants.threads import *
from server.schemas.assistants.assistants import AssistantObject
from server.schemas.assistants.messages import *
from server.schemas.assistants.runs import RunObject
from server.schemas.assistants.streaming import append_message_delta, unwrap_async_queue
from server.schemas.assistants.threads import ThreadObject
from .chat_prompts import prompt_formats
from .context_manager import get_interface,TransformersInterface

prompt_formats_list = list(prompt_formats.keys())


from ..args import *

class ExllamaInferenceContext(ThreadContext):
    job = None
    def __init__(self, run: RunObject, args: ConfigArgs = default_args) -> None:
        super().__init__(run,args)
        

    async def get_local_messages(self):
        input_ids = self.interface.tokenizer.single_token(self.interface.tokenizer.bos_token_id)
        if self.assistant is not None: #拼 prefix prompt
            input_ids = torch.cat([input_ids, self.assistant.get_encoded_instruction(self.encode)],dim=-1)
        
        total_file_count = 0
        parsed_file_count = 0
        # 原本这边的代码拼接了文件内容
        yield parsed_file_count,total_file_count
        # 这里是拼文本重新prefill，应该改成只拼当前用户发送的问题
        for m in self.messages:
            async for ids in m.get_encoded_content(self.encode):
                if isinstance(ids,torch.Tensor):
                    input_ids = torch.cat([input_ids,ids],dim=-1)
                else:
                    parsed_file_count+=1
                    yield parsed_file_count,total_file_count
            
        new_start = "<|im_start|>assistant\n"
        input_ids = torch.cat([input_ids,self.interface.tokenizer.encode(new_start,encode_special_tokens=True)],dim=-1) 
        logger.debug(f'input id length {input_ids.shape[-1]}')
        yield input_ids
        
    def format_message(self, content,role: Role):
        message_start = '<|im_start|>'
        message_end = '<|im_end|>'
        re = f'{message_start}{role}\n{content}{message_end}\n'
        return re
    
    def format_reply(self,content:str):
        message_start = '<|im_start|>'
        message_end = '<|im_end|>'
        content.replace(message_start,'')
        content.replace(message_end,'')
        return content   
    def encode(self,content,role:Role):
        return self.interface.tokenizer.encode(self.format_message(content,role.value),encode_special_tokens=True)
    
    async def work(self):
        logger.debug('start working')
        class JobStage(Enum):
            started = 'started'
            prefill = 'prefill'
            streaming = 'streaming'   

        class Result(BaseModel):
            stage: JobStage
            serial : int


        async for x in self.get_local_messages():
            if isinstance(x,torch.Tensor):
                input_ids = x
            else:
                curr_progress,max_progress = x
                # logger.debug(f'{curr_progress},{max_progress}')
                yield {'stage':'parse','curr_progress':curr_progress,'max_progress':max_progress}
                
        self.job,self.queue = self.interface.create_job(input_ids)
        self.job.related_thread_id = self.thread.id if self.thread.is_related_threads else None
        yield self.run.stream_response_with_event(RunObject.Status.queued) # 给run赋一个状态

        await self.interface.start_doing_job(self.job) # 这个函数在内部运行了prefill和generate


        response_text = ""
        response_str_count = 0
        async for r in unwrap_async_queue(self.queue):
            # r = await self.queue.get()
            # logger.debug(f'get results {r}')
            rs: Result = Result.model_validate(r)
            match rs.stage:
                case JobStage.started:
                    logger.info(f'run {self.run.id} job started')
                    yield self.run.stream_response_with_event(RunObject.Status.in_progress)
                case JobStage.prefill:
                    logger.debug(f'run {self.run.id} prefilling {r}')
                    curr_progress:int = r['curr_progress']
                    max_progress:int = r['max_progress']
                    logger.info(f'run {self.run.id} prefill {curr_progress}/{max_progress}')
                    yield r
                    if (self.job.related_thread_id is not None) and curr_progress==max_progress:
                        break

                case JobStage.streaming:
                    logger.debug(f'run {self.run.id} streaming {r}')
                    if response_str_count==0:
                        
                        user_message = self.messages[-1]
                        if not user_message.role.is_user():
                            raise request_error('user must talk before LLM can talk')
                        user_message.status = MessageObject.Status.completed
                        user_message.sync_db()

                        reply_message = MessageDatabaseManager.create_message_object(
                            self.thread.id,
                            self.run.id,
                            MessageCreate(role=Role.assistant, content=""),    
                        ) 
                        reply_message.status = MessageObject.Status.in_progress
                        reply_message.assistant_id = self.assistant.id
                        reply_message._encoded_content = torch.empty((1,0),dtype=torch.long)
                        self.messages.append(reply_message)

                    if r['eos']:
                        reason = r['eos_reason']
                        logger.info(f'run {self.run.id} job ended, {reason}')
                        reply_message._encoded_content =  self.encode(self.format_reply(r['full_completion']),Role.assistant)
                        break

                    if r['eos'] == False and 'text' not in r:
                        logger.warn('What happened?')
                        continue

                    text:str = r['text']
                    token_ids:torch.Tensor = r['token_ids']
                    response_text += text

                    # replied token ids might be wrong, so we encode them in the end
                    # reply_message._encoded_content = torch.cat([reply_message._encoded_content,token_ids],dim=-1)
                    yield reply_message.append_message_delta(text)
                    response_str_count+=1

             
                case _:
                    raise ValueError

        if self.job.related_thread_id is None:
            logger.info(f"Response: {response_text}")
            reply_message.status = MessageObject.Status.completed
            reply_message.sync_db()

        self.run.status = RunObject.Status.completed
        self.runs_manager.db_sync_run(self.run)
        yield self.run.stream_response_with_event(RunObject.Status.completed)
