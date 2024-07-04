from time import time
from uuid import uuid4
from fastapi import APIRouter
from fastapi.requests import Request
from fastapi.responses import StreamingResponse
from server.backend.context_manager import get_interface ,BackendInterface
from server.schemas.assistants.streaming import chat_stream_response 
from server.schemas.base import ObjectID
from server.schemas.endpoints.chat import ChatCompletionCreate,ChatCompletionChunk,ChatCompletionObject


router = APIRouter()


@router.post('/chat/completions',tags=['openai'])
async def chat_completion(request:Request,create:ChatCompletionCreate):
    id = str(uuid4())

    interface:BackendInterface = get_interface()
    input_ids = interface.format_and_tokenize_input_ids(id,messages=create.get_tokenizer_messages())

    if create.stream:
        async def inner():
            chunk = ChatCompletionChunk(id=id,object='chat.completion.chunk',created=int(time()))
            async for token in interface.work(id,input_ids):     
                chunk.set_token(token)
                yield chunk
        return chat_stream_response(request,inner())
    else:
        comp = ChatCompletionObject(id=id,object='chat.completion.chunk',created=int(time()))
        async for token in interface.work(id,input_ids):     
            comp.append_token(token)
        return comp
