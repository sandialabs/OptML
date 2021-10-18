import pyomo.environ as pyo
from pyomo.core.base.var import ScalarVar, IndexedVar

pyomo_activations = {
    'tanh': lambda xi,*x: pyo.tanh(xi),
    'sigmoid': lambda xi,*x: 1 / (1 + pyo.exp(-xi)),
    'softplus': lambda xi,*x: pyo.log(pyo.exp(xi) + 1),
    'softmax': lambda xi,*x: pyo.exp(xi) / sum(pyo.exp(xj) for xj in x)
}

#pyomo_vector_activations = {'softmax': lambda xj,x: pyo.exp(xj) / sum(pyo.exp(xj) for xi in x)}

def _extract_var_data(vars):
    if isinstance(vars, ScalarVar):
        return [vars]
    elif isinstance(vars, IndexedVar):
        if vars.indexed_set().is_ordered():
            return list(vars.values())
        raise ValueError('Expected IndexedVar: {} to be indexed over an ordered set.'.format(vars))
    elif isinstance(vars, list):
        # Todo: the above if should check if the item supports iteration rather than only list?
        varlist = list()
        for v in vars:
            if v.is_indexed():
                varlist.extend(v.values())
            else:
                varlist.append(v)
        return varlist
    else:
        raise ValueError("Unknown variable type {}".format(vars))
