import json
from time import time
from uuid import uuid4
from fastapi import APIRouter
from fastapi.requests import Request
from server.config.config import Config
from server.backend.context_manager import get_interface,BackendInterface
from server.schemas.assistants.streaming import chat_stream_response, check_link_response, stream_response 
from server.schemas.base import ObjectID
from server.schemas.legacy.completions import CompletionCreate,CompletionObject

router = APIRouter()

@router.post("/completions",tags=['openai'])
async def create_completion(request:Request,create:CompletionCreate):
    id = str(uuid4())

    interface:BackendInterface = get_interface()
    print(f'COMPLETION INPUT:----\n{create.prompt}\n----')

         

    if create.stream:
        async def inner():
            async for token in interface.work(create.prompt,id):     
                d = {'choices':[{'delta':{'content':token}}]}
                yield f"data:{json.dumps(d)}\n\n"
            d = {'choices':[{'delta':{'content':''},'finish_reason':''}]}
            yield f"data:{json.dumps(d)}\n\n"
        return stream_response(request,inner())
    else:
        comp = CompletionObject(id=id,object='text_completion',created=int(time()))
        async for token in interface.work(create.prompt,id):     
            comp.append_token(token)
        return comp
    

