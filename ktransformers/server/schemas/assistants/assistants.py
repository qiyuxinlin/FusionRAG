from time import time
from typing import AsyncIterable, Callable, Dict, List, Optional, Union

from pydantic import BaseModel, Field, PrivateAttr, constr, field_validator, model_validator, validator
import torch

from ktransformers.server.config.config import Config
from ktransformers.server.schemas.assistants.messages import Role
from ktransformers.server.schemas.assistants.runs import *

from ..base import *
from .tool import *

from asyncio import Lock, Queue


class AssistantBase(BaseModel):
    name: Optional[str] = Field(None,description='The name of the assistant.') 
    description: Optional[str] = Field(None,description='The description of the assistant.')
    instructions: Optional[str] = Field(None,description='Instructions which is added in front of the input of LLM') 
    tools: List[Tool] = Field([], max_length=128)

    @field_validator('tools', mode='before')
    def validate_tools(cls, value):
        re = []
        if not isinstance(value, list):
            raise ValueError('Invalid type for tools')

        for t in value:
            if 'type' not in t:
                raise ValueError('Invalid type for tools')
            if t['type'] == 'code_interpreter':
                re.append(CodeInterpreter(**t))
            elif t['type'] == 'file_search':
                re.append(FileSearch(**t))
            elif t['type'] == 'related_threads':
                re.append(RelatedThreads(**t))
            elif t['type'] == 'function':
                re.append(FuntionTool(**t))
            else:
                raise ValueError('Invalid type for tools')
        return re

    tool_resources: List[ToolResource] = Field([], max_length=128)

    @field_validator('tool_resources', mode='before')
    def validate_tool_resources(cls, value):
        re = []
        if not isinstance(value, list):
            raise ValueError('Invalid type for tool resources')

        for tr in value:
            if 'file_ids' in tr:
                re.append(CodeInterpreterResource(**tr))
            elif 'vector_stores' in tr:
                re.append(FileSearchResource(**tr))
            elif 'thread_ids' in tr:
                re.append(RelatedThreadsResource(**tr))
            else:
                raise ValueError('Invalid type for tool resources')
        return re

    meta_data: Metadata = MetadataField

    @model_validator(mode='before')
    def convert_meta_data(cls, values):
        if 'meta_data' in values:
            values['metadata'] = values['meta_data']
        return values
    temperature: Optional[float] = Field(ge=0.0, le=2.0, default=1)
    top_p: Optional[float] = Field(ge=0.0, le=1.0, default=1)
    response_format: Union[str, Dict[str, str]] = "auto"


class AssistantCreate(AssistantBase):
    model: str


class AssistantBuildStatus(BaseModel):
    class Status(Enum):
        not_build = "not_build"
        in_queue = "in_queue"
        parsing = "parsing"
        prefilling = "prefilling"
        dumping = "dumping"
        completed = "completed"
        paused = "paused"

    _lock: Lock = PrivateAttr(default_factory=Lock)
    _queue: Optional[Queue] = PrivateAttr(None)

    status: Status = Field(default=Status.not_build)
    total_file_count: int = Field(default=0)
    parsed_file_count: int = Field(default=0)

    prefilling_current: int = Field(default=0)
    prefilling_total: int = Field(default=0)

    build_started_time: Optional[int] = Field(default=None)
    build_completed_time: Optional[int] = Field(default=None)

    # in megabytes
    assistant_usage: int = Field(default=0, description='')
    assistant_total_usage: int = Field(default=0)
    disk_free_space: int = Field(default=0)
    disk_total_space: int = Field(default=0)

    def to_stream_reply(self) -> str:
        return f"event: assistant.build.status\ndata: {self.model_dump_json()}\n\n"


class AssistantObject(AssistantBase, ObjectWithCreatedTime):
    model: Optional[str] = Field(
        default=Config().model_name)
    related_threads_objects: Optional[List] = Field(None, exclude=True)
    _encoded_instruction: Optional[torch.Tensor] = PrivateAttr(default=None)
    build_status: AssistantBuildStatus = Field(default=AssistantBuildStatus())

    def as_api_response(self):
        return self.model_dump(exclude={'build_status'})

    def get_related_threads_ids(self) -> List[ObjectID]:
        re = []
        for t, tr in zip(self.tools, self.tool_resources):
            if t.type == ToolType.RELATED_THREADS:
                re += tr.thread_ids or []
        return re

    def get_related_threads_objects(self) -> List:
        raise NotImplementedError  # should be replaced

    def append_related_threads(self, thread_ids: List[ObjectID]):
        # logger.debug(f'{self.tools} {self.tool_resources}')
        for t, tr in zip(self.tools, self.tool_resources):
            if t.type == ToolType.RELATED_THREADS:
                tr.thread_ids += thread_ids
                return

        self.tools.append(RelatedThreads(type=ToolType.RELATED_THREADS))
        self.tool_resources.append(
            RelatedThreadsResource(thread_ids=thread_ids))

    async def update_build_status(self, events: AsyncIterable) -> AsyncIterable:
        async for event in events:
            # logger.debug(event)
            if isinstance(event, RunStreamResponse):
                if event.event == RunObject.Status.completed:
                    self.build_status.status = AssistantBuildStatus.Status.completed
                    self.build_status.build_completed_time = int(time())
                    self.sync_db()
                    yield self.build_status.model_copy()
            elif isinstance(event, dict):
                # logger.debug('dict')
                if 'stage' in event:
                    if event['stage'] == 'prefill':
                        self.build_status.status = AssistantBuildStatus.Status.prefilling
                        self.build_status.prefilling_current = event['curr_progress']
                        self.build_status.prefilling_total = event['max_progress']
                    if event['stage'] == 'parse':
                        self.build_status.status = AssistantBuildStatus.Status.parsing
                        self.build_status.parsed_file_count = event['curr_progress']
                        self.build_status.total_file_count = event['max_progress']
                    yield self.build_status.model_copy()

    def get_build_status(self) -> AssistantBuildStatus:
        return self.build_status
     
    
    def sync_db(self)->None:
        raise NotImplementedError # should be replaced
    
    def get_encoded_instruction(self,encode_fn:Callable)->torch.Tensor:
        if self._encoded_instruction is None:
            logger.info(f'encoding assistant instruction: {self.instructions}')
            self._encoded_instruction = encode_fn(self.instructions, Role.user)
        return self._encoded_instruction


class AssistantModify(AssistantBase):
    model: Optional[str] = None


# Non API Backend
