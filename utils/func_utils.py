from ast import literal_eval
def isevaluatable(s):
    import ast
    try:
        literal_eval(s)
        return True
    except ValueError:
        return False
def transform_group(group):
    for column in ['new_hosp_yhat', 'new_death_yhat']:
        min_el = group[column].apply(lambda x: x if x>0 else None).dropna()
        if len(min_el) == 0:
            min_el = 0
        else:
            min_el = min_el.min()
        group[column] = group[column].apply(lambda x: x if x >= 0 else min_el)
    return group