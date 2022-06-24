from pprint import pprint
from typing import List


class Log:
    def __init__(self, name=None, types=None, content=None, message=None):
        self.name = name
        self.types = types if types is not None else []
        self.content = content
        self.message = message

    def __str__(self):
        return str(self.content)


class TypedLog:

    def __init__(self):
        self.logs: List[Log] = []

    def new_log_obj(self, name=None, types=None, content=None, message=None):
        log = Log(name=name, types=types, content=content, message=message)
        self.logs.append(log)
        return log

    def new_log_content(self, name=None, types=None, content=None, message=None):
        log = self.new_log_obj(name=name, types=types, content=content, message=message)
        return log.content

    def get_log_by_name(self, prefix):
        for log in self.logs[::-1]:
            if log.name.startswith(prefix):
                return log
        raise Exception(f"Log with prefix {prefix} is not found")

    def get_log_by_type(self, type_name):
        for log in self.logs[::-1]:
            if type_name in log.types:
                return log
        raise Exception(f"Log with type {type_name} is not found")

    def show(self):
        for log in self.logs.items():
            print(f"Name: {log.name}")
            if log.message is not None:
                print(f"Message: {log.message}")
            if len(log.types) > 0:
                print(f"Type: {log.types}")
            pprint(log.content)
            print()
