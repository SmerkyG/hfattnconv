from dataclasses import dataclass
import datetime
import typing

class CLIError(ValueError):
    pass

class Config(dict):
    def __init__(self, **kwargs):
        for name, value in kwargs:
            self[name] = value
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(e)
    def __setattr__(self, name, value):
         self[name] = value

def convert_dict_to_config(src:dict):
    out = Config()
    for srckey, srcval in src.items():
        if isinstance(srcval, dict):
            out[srckey] = convert_dict_to_config(srcval)
        else:
            out[srckey] = srcval
    return out

def merge_config(dst:dict, src:dict):
    for key, srcval in src.items():
        if key in dst.keys() and isinstance(srcval, dict) and isinstance(dst[key], dict):
            merge_config(dst[key], srcval)
        else:
            dst[key] = srcval
    return dst

import ast
from ast import Constant, UnaryOp, USub

def literal_eval(s:str):
    def _convert_num(node):
        if not isinstance(node, Constant) or type(node.value) not in (int, float, complex):
            raise ValueError()
        return node.value

    # if we can convert to a constant, great
    # if not, treat it as a string
    try:
        node = ast.parse(s.lstrip(" \t"), mode='eval')
        if isinstance(node, ast.Expression):
            node = node.body
        multiplier = 1
        if isinstance(node, UnaryOp) and isinstance(node.op, USub):
            multiplier = -1
            node = node.operand
        return multiplier * _convert_num(node)
    except:
        pass
    s = s.encode('latin-1','backslashreplace').decode('unicode_escape')
    return s

def parse_args(args:list, out:Config|None = None):
    if len(args) % 2 != 0:
        raise CLIError("bad number of arguments (they must all be pairs)")
    if out is None:
        out = Config()
    for name, value in zip(args[0::2], args[1::2]):
        if not name.startswith('--'):
            raise CLIError(f'argument names must start with -- but {name} did not')
        name = name[2:]
        parts = name.split('.')
        subobj = out
        for part in parts[:-1]:
            if not hasattr(subobj, part):
                subobj[part] = Config()
            subobj = subobj[part]
        try:
            subobj[parts[-1]] = literal_eval(value)
        except:
            raise CLIError(f"could not parse value for '{name}': {value}")
    return out

import yaml
import json
import re

# bugfix for yaml parsing of floats without period
yaml_re = re.compile(u'''^(?:
    [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
    |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
    |\\.[0-9_]+(?:[eE][-+][0-9]+)?
    |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
    |[-+]?\\.(?:inf|Inf|INF)
    |\\.(?:nan|NaN|NAN))$''', re.X)
yaml_loader = yaml.SafeLoader
yaml_loader.add_implicit_resolver(
    u'tag:yaml.org,2002:float',
    yaml_re,
    list(u'-+0123456789.'),
    )

def load_configs(paths:list[str], out:Config|None = None):
    if out is None:
        out = Config()

    for path in paths:
        if path.endswith('.yaml'):
            with open(path, mode="rt", encoding="utf-8") as file:
                config = yaml.load(file, yaml_loader)
        elif path.endswith('.json'):
            with open(path, mode="rt", encoding="utf-8") as file:
                config = json.load(file)
        else:
            raise ValueError(f"only .yaml and .json config files are supported, but got `{path}`")
        config = convert_dict_to_config(config)
        out = merge_config(out, config)
    return out

import inspect
from pydoc import locate

def type_name(t):
    if type(t) != type and not callable(t):
        return str(t)
    origin = typing.get_origin(t)
    if origin is None:
        return ((t.__module__ + '.') if t.__module__ != 'builtins' else '') + t.__qualname__
    else:
        rv = type_name(origin) + '['
        for arg in typing.get_args(t):
            rv += type_name(arg)
        rv += ']'
        return rv

def typecheck(path : str, obj : typing.Any, required_type : type = typing.Any, parent : typing.Any = None, key:str = ''):
    errors = ''
    try:
        if isinstance(required_type, str):
            # if the required type is a string (which could happen due to weird python pre-declarations that aren't available, and is done in lightning.LightningModule.fit's model parameter type)
            # then just allow anything through, since we can't realistically type check this
            required_type = typing.Any

        if isinstance(obj, Config):
            if required_type == typing.Any:
                pass
            else:
                if '__type__' in obj:
                    sub_required_type = locate(obj['__type__'])
                    if sub_required_type is None:
                        return f'Explicit object type {obj["__type__"]} not found (did you forget to specify the module?)\n' 
                    if not issubclass(sub_required_type, required_type):
                        return f'Explicit object type {obj["__type__"]} specified for `{path}` is not compatible with {required_type}\n' 
                    required_type = sub_required_type

                sig = inspect.signature(required_type.__init__)
                for k in obj.keys():
                    if k != '__type__' and k not in sig.parameters.keys():
                        return f'Disallowed config entry `{path}.{k}` - No such parameter `{k}` in {required_type}\n'
                    
                # traverse all subelements recursively
                for k, param in sig.parameters.items():
                    if k == 'self':
                        continue
                    if k in obj.keys():
                        rt = param.annotation
                        if rt == inspect.Parameter.empty:
                            rt = typing.Any
                        errors += typecheck(k if path == '' else path + '.' + k, obj[k], rt, obj, k)
                    elif param.default == inspect.Parameter.empty:               
                        return f'Required parameter `{path}.{k}` missing in {required_type}\n'
                    else:
                        # add default value into config
                        obj[k] = param.default

        elif required_type != typing.Any and not isinstance(obj, required_type) and not (required_type == float and isinstance(obj, int)):
            if required_type == str and not isinstance(obj, str):
                # allow conversion to string, if we wanted a string
                parent[key] = str(obj)
            else:
                errors += f"Config Type Mismatch: expected {type_name(required_type)} but got {type_name(type(obj))}\n in config setting `{path}` : {type_name(required_type)} = {obj}\n"
                return errors

    except Exception as ex:
        raise Exception(f'internal config type checking exception at path "{path}": {required_type} {ex}')

    return errors            

def parse_cmdline_configs(argv, base_config_type:type):
    argv = argv.copy()
    config_paths = []
    i = 0
    while i < len(argv):
        if argv[i] == '-c':
            argv.pop(i)
            config_paths.append(argv.pop(i))
        else:
            i += 2
    config : base_config_type = load_configs(config_paths)
    config = parse_args(argv, config)
    errors = typecheck('', config, base_config_type)   
    return config, errors

if __name__ == '__main__':
    @dataclass(kw_only=True)
    class CLI_Config:
        path: str
        seed: int | None = None
        recurrent: int = 1
        train : typing.Any = None
        model: Model_Config

    import sys
    config, errors = parse_cmdline_configs(sys.argv[1:], CLI_Config)
    print(config)
    if errors != '':
        print(errors)
