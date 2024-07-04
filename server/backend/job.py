import sys, os
from typing import Dict, Tuple
import torch.nn as nn

from .text_streamer import TextStreamer
sys.path.append(
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "exllamav2",
    )
)

import torch
from asyncio import Queue

from server.utils.multi_timer import MultiTimer
from enum import Enum

from server.crud.assistants.messages import MessageDatabaseManager
from ..crud.assistants.assistants import AssistantDatabaseManager
from ..crud.assistants.runs import RunsDatabaseManager
from ..crud.assistants.threads import *
from ..schemas.assistants.assistants import AssistantObject
from ..schemas.assistants.messages import *
from ..schemas.assistants.runs import RunObject
from ..schemas.assistants.streaming import append_message_delta, unwrap_async_queue
from ..schemas.assistants.threads import ThreadObject
from .chat_prompts import prompt_formats
from .context_manager import get_interface,TransformersInterface

prompt_formats_list = list(prompt_formats.keys())


from .args import *



class inferenceContext:
    # Assistant Logic
    assistant: Optional[AssistantObject] = None
    related_threads : List[ThreadObject]
    
    thread: ThreadObject

    messages: List[MessageObject] = []  # messages must be user, system, user, system
    run: RunObject

    # Inference Context
    interface: Optional[TransformersInterface] = None
    
    ever_prepared: bool = False
    args: ConfigArgs
    system_prompt: Any
    prompt_format: Any
    
    queue: Optional[Queue] = None
    timer: MultiTimer = MultiTimer()

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
        self.interface = get_interface()
        self.update_by_run(run,args)

    def update_by_run(self,run:RunObject,args:ConfigArgs = default_args):
        self.run = run
        
        self.args = args
        self.prompt_format = prompt_formats[args.mode]()
        self.prompt_format.botname = args.botname
        self.prompt_format.username = args.username
        self.system_prompt = self.prompt_format.default_system_prompt()

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
        raise NotImplementedError
        


    def put_user_message(self, message: MessageObject):
        assert (
            message.role.is_user() and message.thread_id == self.thread.id and message.status == MessageObject.Status.in_progress
        )
        self.messages.append(message)

    def delete_user_message(self,message_id: ObjectID):
        self.messages = [m for m in self.messages if m.id != message_id]


    async def get_local_messages(self):
        raise NotImplementedError


    async def work(self):
        raise NotImplementedError
class TransformersInferenceContext(inferenceContext):  

    def __init__(self, run: RunObject, args: ConfigArgs = default_args) -> None:
        super().__init__(run,args)
    

    def get_local_messages(self):
        local_messages = []
        for m in self.messages:
            local_messages.append(
                {'role':m.role.value,
                 'content':m.get_text_content()}
            )
        
        return local_messages


   
    async def work(self):
        logger.debug('start working')
        input_ids = self.interface.format_and_tokenize_input_ids(self.thread.id, self.get_local_messages())
        user_message = self.messages[-1]
        if not user_message.role.is_user():
            raise request_error('user must talk before LLM can talk')
        user_message.status = MessageObject.Status.completed
        user_message.sync_db()


        response_str_count = 0  
        reply_message = self.message_manager.create_message_object(
                            self.thread.id,
                            self.run.id,
                            MessageCreate(role=Role.assistant, content=""),    
                        )
        reply_message.assistant_id = self.assistant.id
        self.messages.append(reply_message) 

        yield reply_message.stream_response_with_event(MessageObject.Status.created)
       
        
        reply_message.append_message_delta(' ') # For Mistra we should firstly append a ' '
        yield reply_message.stream_response_with_event(MessageObject.Status.in_progress)

        yield self.run.stream_response_with_event(RunObject.Status.in_progress)

        async for token in self.interface.work(self.thread.id,input_ids):     
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
        

class ExllamaInferenceContext(inferenceContext):
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
