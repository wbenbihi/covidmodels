from ast import literal_eval

def isevaluatable(s):
    import ast
    try:
        literal_eval(s)
        return True
    except ValueError:
        return False