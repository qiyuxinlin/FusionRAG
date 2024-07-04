#!/usr/bin/env python
# coding=utf-8
'''
Description  : Implement singleton
Author       : unicornchan
Date         : 2024-06-11 17:08:36
Version      : 1.0.0
LastEditors  : unicornchan 
LastEditTime : 2024-06-12 08:25:54

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
import abc

class Singleton(abc.ABCMeta, type):
    """_summary_

    Args:
        abc.ABCMeta: Provide a mechanism for defining abstract methods and properties,
            enforcing subclasses to implement these methods and properties.
        type: Inherit from 'type' to make 'Singleton' a metaclass,
            enabling the implementation of the Singleton
    """
    _instances = {}

    def __call__(cls, *args, **kwds):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwds)
        return cls._instances[cls]

class AbstractSingleton(abc.ABC, metaclass=Singleton):
    """Provided an abstract Singleton base class, any class inheriting from
       this base class will automatically become a Singleton class.

    Args:
        abc.ABC: Abstract base class, it cannot be instantiated, only inherited. 
    """
