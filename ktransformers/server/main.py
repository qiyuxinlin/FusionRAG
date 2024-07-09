import sys
import os
import re
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import uvicorn.logging
project_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(project_dir)
import argparse
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from server.config.config import Config
from server.backend.context_manager import globalInterface,BackendInterface
from server.backend.args import default_args
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
    cfg = Config()
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
    if cfg.mount_web:
        mount_index_routes(app)
    return app

def update_web_port(config_file: str):
    ip_port_pattern = r"(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?):([0-9]{1,5})"
    with open(config_file, "r", encoding="utf-8") as f_cfg:
        web_config = f_cfg.read()
    print(web_config)
    ip_port = "localhost://" + str(Config().server_port)
    new_web_config = re.sub(ip_port_pattern, ip_port, web_config)
    with open(config_file, "w", encoding="utf-8") as f_cfg:
        f_cfg.write(new_web_config)


def mount_index_routes(app: FastAPI):
    web_dir = os.path.join(project_dir, "website/dist")
    web_config_file = os.path.join(web_dir, "config.js")
    update_web_port(web_config_file)
    if os.path.exists(web_dir):
        app.mount("/web", StaticFiles(directory=web_dir), name="static")
    else:
        err_str = f"No website resources in {web_dir}, please complile the website by npm first"
        logger.error(err_str)
        print(err_str)
        exit(1)


def run_api(app, host, port, **kwargs):
    if kwargs.get("ssl_keyfile") and kwargs.get("ssl_certfile"):
        uvicorn.run(app,
                    host=host,
                    port=port,
                    ssl_keyfile=kwargs.get("ssl_keyfile"),
                    ssl_certfile=kwargs.get("ssl_certfile"),
                    )
    else:
        uvicorn.run(app, host=host, port=port, log_level='debug')


def main():
    cfg = Config()
    parser = argparse.ArgumentParser(prog='Approaching.AI',
                                     description='Lexllama: Efficient Long Context Inference')
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9016)
    parser.add_argument("--ssl_keyfile", type=str)
    parser.add_argument("--ssl_certfile", type=str)
    parser.add_argument("--web", type=bool, default=False)
    parser.add_argument("--model_name", type=str, default=cfg.model_name)
    parser.add_argument("--model_path", type=str, default=cfg.model_path)
    parser.add_argument("--device", type=str, default=cfg.model_device)
    parser.add_argument("--gguf_path", type=str, required=False)
    parser.add_argument("--optimize_config_path", type=str, required=False)

    # 初始化消息
    args = parser.parse_args()
    cfg.model_name = args.model_name
    cfg.model_path = args.model_path
    cfg.model_device = args.device
    cfg.mount_web = args.web
    cfg.server_ip = args.host
    cfg.server_port = args.port

    default_args.model_dir = args.model_path
    default_args.device = args.device
    default_args.gguf_path = args.gguf_path
    default_args.optimize_config_path = args.optimize_config_path
    app = create_app()
    globalInterface.interface = BackendInterface(default_args)
    run_api(app=app,
            host=args.host,
            port=args.port,
            ssl_keyfile=args.ssl_keyfile,
            ssl_certfile=args.ssl_certfile,)


if __name__ == "__main__":
    main()
