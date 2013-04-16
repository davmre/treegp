
import types, marshal

def marshal_fn(f):
    if f.func_closure is not None:
        raise ValueError("function has non-empty closure %s, cannot marshal!" % repr(f.func_closure))
    s = marshal.dumps(f.func_code)
    return s

def unmarshal_fn(dumped_code):
    f_code = marshal.loads(dumped_code)
    f = types.FunctionType(f_code, globals())
    return f
