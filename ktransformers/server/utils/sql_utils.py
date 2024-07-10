#!/usr/bin/env python
# coding=utf-8
'''
Description  :  
Author       : unicornchan
Date         : 2024-06-12 09:12:58
Version      : 1.0.0
LastEditors  : unicornchan 
LastEditTime : 2024-06-16 15:49:58

The MIT License (MIT)
Copyright (c) 2024  by Approach.AI
Permission is hereby granted, free of charge, to any person obtaining a copy of this
software and associated documentation files (the “Software”), to deal in the Software
without restriction, including without limitation the rights to use, copy, modify, 
merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all copies
or substantial portions of the Software. The Software is provided “as is”, without warranty
of any kind, express or implied, including but not limited to the warranties of merchantability,
fitness for a particular purpose and noninfringement. In no event shall the authors or
copyright holders be liable for any claim, damages or other liability, whether in an
action of contract, tort or otherwise, arising from, out of or in connection with the
software or the use or other dealings in the Software.

'''

from urllib.parse import urlparse
import os
from contextlib import contextmanager
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker, declarative_base

from ktransformers.server.config.config import Config
from ktransformers.server.config.singleton import Singleton
from ktransformers.server.config.log import logger
from ktransformers.server.exceptions import db_exception


Base = declarative_base()


class SQLUtil(metaclass=Singleton):
    """
    database connections init and management
    """
    sqlalchemy_engine = None
    session_local = None

    def __init__(self) -> None:
        self.cfg: Config = Config()
        if not self.sqlalchemy_engine:
            SQLUtil.init_engine(self.cfg)

    @contextmanager
    def get_db(self):
        """
        After you finish using the session, it's crucial to close it.
        """
        if not SQLUtil.sqlalchemy_engine:
            SQLUtil.init_engine(self.cfg)
        session = self.session_local()  # type: ignore pylint: disable=not-callable
        try:
            yield session
        finally:
            session.close()

    @staticmethod
    def init_engine(cfg: Config):
        """
        initial engine and session maker Factory
        """
        pool_size = cfg.db_pool_size
        if SQLUtil.sqlalchemy_engine is None:
            if cfg.db_type == "sqllite":
                db_url = SQLUtil.create_sqllite_url(cfg)
            else:
                logger.error("Unsupported database type %s", cfg.db_type)
                exit(-1)
            SQLUtil.sqlalchemy_engine = create_engine(
                db_url, connect_args={"check_same_thread": False}, pool_size=pool_size)
            SQLUtil.session_local = sessionmaker(
                autocommit=False, autoflush=False, bind=SQLUtil.sqlalchemy_engine)

    @staticmethod
    def create_sqllite_url(cfg):
        """
        create and validate SQLLite url
        """
        path: str = cfg.db_host
        database: str = cfg.db_database
        absolute_path: str = os.path.join(path, database)
        url = 'sqlite:///' + absolute_path
        try:
            result = urlparse(url)
            if all([result.scheme, result.path, result.scheme == 'sqlite']):
                return url
            else:
                logger.error("invalid sqllite url: %s", url)
                exit(-1)
        except ValueError:
            logger.error("invalid sqllite url: %s", url)
            exit(-1)

    def db_add_commit_refresh(self, session: Session, what):
        """
        add data to database
        """
        try:
            session.add(what)
            session.commit()
            session.refresh(what)
        except Exception as e:
            logger.exception("db commit error with data %s", str(what.__dict__))
            ex = db_exception()
            ex.detail = str(e)
            session.rollback()
            raise ex from e

    def db_merge_commit(self, session: Session, what):
        try:
            session.merge(what)
            session.commit()
        except Exception as e:
            ex = db_exception()
            ex.detail = str(e)
            logger.exception("db merge commit error with data %s", str(what.__dict__))
            session.rollback()
            raise ex from e

    def db_update_commit_refresh(self, session: Session, existing, what):
        what = what.model_dump(mode="json")
        try:
            for key in what.keys():
                if what[key] is not None:  # 检查b中的字段是否为None
                    setattr(existing, key, what[key])  # 更新a的字段
            session.commit()
            session.refresh(existing)
        except Exception as e:
            ex = db_exception()
            ex.detail = str(e)
            logger.exception("db update commit refresh error with data %s", str(what.__dict__))
            session.rollback()
            raise ex from e
