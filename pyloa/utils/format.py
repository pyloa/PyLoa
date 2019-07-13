def str_iterable(iterable, depth=0, delimiter='    ', markdown_newline=False):
    """a helper function converting nested, iterable containers to str representation with visual indents for logging"""
    msg = "" if markdown_newline is False else "    "
    # no visual indent for len==0 iterables
    if len(iterable) == 0:
        if isinstance(iterable, list):
            return "[]"
        if isinstance(iterable, tuple):
            return "()"
        if isinstance(iterable, dict):
            return "{}"
    # do not nest lists or tuples with visual indents
    if isinstance(iterable, (list, tuple)):
        msg += "[\n"
        for value in iterable:
            msg += delimiter*(depth+1)
            if isinstance(value, (list, tuple, dict)):
                msg += "{},\n".format(str_iterable(value, depth+1))
            else:
                msg += "{},\n".format(str(value))
        msg += delimiter * depth + "]"
        return msg
    # nested visual indents for dictionaries
    if isinstance(iterable, dict):
        msg += "{\n"
        for key, value in iterable.items():
            msg += "{}{}: ".format(delimiter*(depth+1), str(key))
            if isinstance(value, (list, tuple, dict)):
                msg += "{},\n".format(str_iterable(value, depth+1))
            else:
                msg += "{},\n".format(str(value))
        msg += delimiter * depth + "}"
        return msg
