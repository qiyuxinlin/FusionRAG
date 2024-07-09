import sys,os

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4'

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import uvicorn.logging
project_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(project_dir)
sys.path.append(os.path.join(os.getcwd(),'inference'))
from configs import VERSION
import argparse
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from server.config.config import Config
from server.backend.context_manager import globalInterface,BackendInterface
from server.backend.args import default_args
from fastapi.openapi.utils import get_openapi

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


from server.api import router, post_db_creation_operations
from server.utils.sql_utils import Base, SQLUtil
from server.config.log import logger


def mount_app_routes(mount_app: FastAPI):
    sql_util = SQLUtil()
    logger.info("Creating SQL tables")
    Base.metadata.create_all(bind=sql_util.sqlalchemy_engine)
    post_db_creation_operations()
    mount_app.include_router(router)


def create_app():
    app = FastAPI()
    if Config().web_cross_domain:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    mount_app_routes(app)
    mount_index_routes(app)
    return app

def mount_index_routes(app: FastAPI):
    # app.mount("/web", StaticFiles(directory=(project_dir + "/website/dist")), name="static")
    pass



def run_api(app, host, port, **kwargs):
    if kwargs.get("ssl_keyfile") and kwargs.get("ssl_certfile"):
        uvicorn.run(app,
                    host=host,
                    port=port,
                    ssl_keyfile=kwargs.get("ssl_keyfile"),
                    ssl_certfile=kwargs.get("ssl_certfile"),
                    )
    else:    
        uvicorn.run(app, host=host, port=port,log_level='debug')

def custom_openapi(app):
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="ktransformers server",
        version="1.0.0",
        summary="This is a server that provides a RESTful API for ktransformers.",
        description="We provided chat completion and openai assistant interfaces.",
        routes=app.routes,
    )
    openapi_schema["info"]["x-logo"] = {
        "url": "https://kvcache.ai/media/icon_1.png"
    }
    app.openapi_schema = openapi_schema
    return app.openapi_schema

def main():
    cfg = Config()
    parser = argparse.ArgumentParser(prog='Approaching.AI',
                                     description='Lexllama: Efficient Long Context Inference')
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9016)
    parser.add_argument("--ssl_keyfile", type=str)
    parser.add_argument("--ssl_certfile", type=str)
    # parser.add_argument("--model_name", type=str, default=cfg.model_name)
    # parser.add_argument("--model_path", type=str, default=cfg.model_path)
    # parser.add_argument("--device", type=str, default=cfg.model_device)
    # parser.add_argument("--gguf_path", type=str, required=False)
    # parser.add_argument("--optimize_config_path", type=str, required=False)
    # 初始化消息


    # 初始化消息
    args = parser.parse_args()
    app = create_app()
    custom_openapi(app)
    globalInterface.interface = BackendInterface(default_args)
    run_api(app=app,
            host=args.host,
            port=args.port,
            ssl_keyfile=args.ssl_keyfile,
            ssl_certfile=args.ssl_certfile,)
    
if __name__ == "__main__":
    main()
