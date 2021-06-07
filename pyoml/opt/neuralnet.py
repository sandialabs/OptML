import pyomo.environ as pyo
import pyomo.mpec as mpec
from pyomo.core.base.var import ScalarVar, IndexedVar
from pyomo.core.base.block import _BlockData, declare_custom_block
from .utils import _keras_sequential_to_dict

#Build the full-space representation of a neural net. This is a module-level function for now.
def build_neural_net(block,n_inputs,n_outputs,n_nodes,w,b):
    #block sets
    block.NODES = pyo.Set(initialize=list(range(n_nodes)), ordered=True)
    block.INTERMEDIATE_NODES = pyo.Set(initialize=list(range(n_inputs, n_nodes)), ordered=True)
    block.INPUTS = list(block.inputs_set)
    block.OUTPUTS = [i + n_nodes for i in list(block.outputs_set)]

    #mapping for inputs
    block.x = dict(zip(block.INPUTS,block.inputs.values()))

    #mapping for outputs
    block.y = dict(zip(block.OUTPUTS,block.outputs.values()))

    # pre-activation values
    block.zhat = pyo.Var(block.INTERMEDIATE_NODES)

    # post-activation values
    block.z = pyo.Var(block.NODES)

    # set inputs
    @block.Constraint(block.INPUTS)
    def _inputs(m,i):
        return m.z[i] == m.x[i]

    # pre-activation logic
    @block.Constraint(block.INTERMEDIATE_NODES)
    def _linear(m,i):
        return m.zhat[i] == sum(w[i][j]*m.z[j] for j in w[i]) + b[i]

    # output logic
    @block.Constraint(block.OUTPUTS)
    def _outputs(m,i):
        return m.y[i] == sum(w[i][j]*m.z[j] for j in w[i]) + b[i]

import abc
class DefinitionInterface(abc.ABC):
    def __init__(self):
        pass

    @abc.abstractmethod
    def n_inputs(self):
        pass

    @abc.abstractmethod
    def n_outputs(self):
        pass

    @abc.abstractmethod
    def build(self, neural_block):
        pass

class NeuralNetInterface(DefinitionInterface):
    def __init__(self):
        self.num_nodes = 0
        self.num_inputs = 0
        self.num_outputs = 0
        self.w = dict()
        self.b = dict()

    def n_inputs(self):
        return self.num_inputs

    def n_outputs(self):
        return self.num_outputs

    @abc.abstractmethod
    def build(self,neural_block):
        pass

    def _check_weights(self):
        if self.w == dict():
            raise ValueError("""No weights have been defined for the neural net definition.  Use set_weights(w,b,n_inputs,n_outputs,n_nodes)
            to define the network.  Alternatively, you may use set_weights_keras(keras_model) if you have en existing
            keras model.""")
        else:
            pass

    def set_weights(self,w,b,n_inputs,n_outputs,n_nodes):
        assert n_nodes >= n_inputs
        assert len(w) == n_nodes + n_outputs - n_inputs
        assert len(b) == n_nodes + n_outputs - n_inputs
        self.w = w
        self.b = b
        self.num_inputs = n_inputs
        self.num_outputs = n_outputs
        self.num_nodes = len(w) - self.num_outputs + self.num_inputs

    def set_weights_keras(self,keras_model):
        """
            Unpack a keras model into dictionaries of weights and biases.  The dictionaries are used to build the underlying pyomo model.
        """
        w,b = _keras_sequential_to_dict(keras_model)  #TODO: sparse version
        n_inputs = len(keras_model.get_weights()[0])
        n_outputs = len(keras_model.get_weights()[-1])
        n_nodes = len(w) - self.num_outputs + self.num_inputs
        self.set_weights(w,b,n_inputs,n_outputs,n_nodes)


class FullSpaceNonlinear(NeuralNetInterface):
    """
        Builder class that creates a neural network surrogate for a Pyomo model.
        This class exposes the intermediate neural network variables as Pyomo variables and constraints.
    """
    def __init__(self,activation = pyo.tanh):
        super().__init__()
        self.activation = activation

    def _build_neural_net(self,block):
        self._check_weights()
        build_neural_net(block,self.num_inputs,self.num_outputs,self.num_nodes,self.w,self.b)

    def _add_activation_constraint(self,block):
        @block.Constraint(block.INTERMEDIATE_NODES)
        def _z_activation(m,i):
            return m.z[i] == self.activation(m.zhat[i])

    def set_activation_func(self,activation_function):
        self.activation = activation

    def build(self,block):
        self._build_neural_net(block)
        self._add_activation_constraint(block)

class ReducedSpaceNonlinear(NeuralNetInterface):
    """
        Builder class that creates a neural network surrogate for a Pyomo model.
        This class hides the intermediate nerual network variables inside Pyomo expressions.
    """
    def __init__(self,activation = pyo.tanh):
        super().__init__()
        self.activation = activation

    def _unpack_nn_expression(self,block,i):
        """
            Creates a Pyomo expression for output `i` of a neural network.  Uses recursion to build up the expression.
        """
        nodes_from = self.w[i]
        z_from = dict()
        for node in nodes_from:
            if node in self.w: #it's an output or intermediate node
                z_from[node] = self._unpack_nn_expression(block,node)
            else:              #it's an input node
                z_from[node] = block.x[node]

        if i in block.OUTPUTS: #don't apply activation to output
            z = sum(self.w[i][j]*z_from[j] for j in self.w[i]) + self.b[i]
        else:
            z = sum(self.w[i][j]*self.activation(z_from[j]) for j in self.w[i]) + self.b[i]

        return z

    def _build_neural_net(self,block):
        self._check_weights()
        block.INPUTS = list(block.inputs_set)
        block.OUTPUTS = [i + self.num_nodes for i in list(block.outputs_set)]

        #mapping for inputs
        block.x = dict(zip(block.INPUTS,block.inputs.values()))

        #mapping for outputs
        block.y = dict(zip(block.OUTPUTS,block.outputs.values()))

        def neural_net_constraint_rule(block,i):
            expr = self._unpack_nn_expression(block,i)
            return block.y[i] == expr

        block._neural_net_constraint = pyo.Constraint(block.OUTPUTS,rule=neural_net_constraint_rule)

    def build(self,block):
        self._build_neural_net(block)


class BigMReLU(NeuralNetInterface):
    """
        Builder class for creating a MILP representation of a
        ReLU or LeakyReLU neural network on a Pyomo model.
    """
    def __init__(self,bigm = 1e6,leaky_alpha = None):
        super().__init__()
        self.M = bigm
        self.leaky_alpha = leaky_alpha
        if leaky_alpha is not None:
            raise NotImplementedError('LeakyReLU is not yet implemented')

    def _build_neural_net(self,block):
        self._check_weights()
        build_neural_net(block,self.num_inputs,self.num_outputs,self.num_nodes,self.w,self.b)

    def _add_activation_constraint(self,block):
        # activation indicator q=0 means z=zhat (positive part of the hinge)
        # q=1 means we are on the zero part of the hinge
        block.q = pyo.Var(block.INTERMEDIATE_NODES, within=pyo.Binary)

        @block.Constraint(block.INTERMEDIATE_NODES)
        def _z_lower_bound(m,i):
            return m.z[i] >= 0

        @block.Constraint(block.INTERMEDIATE_NODES)
        def _z_zhat_bound(m,i):
            return m.z[i] >= m.zhat[i]

        # These are the activation binary constraints
        @block.Constraint(block.INTERMEDIATE_NODES)
        def _z_hat_positive(m,i):
            return m.z[i] <= m.zhat[i] + self.M*m.q[i]

        @block.Constraint(block.INTERMEDIATE_NODES)
        def _z_hat_negative(m,i):
            return m.z[i] <= self.M*(1.0-m.q[i])


    def build(self,block):
        self._build_neural_net(block)
        self._add_activation_constraint(block)

    #TODO
    def _perform_bounds_tightening(self,block):
        pass

    def _build_relaxation(self,block):
        pass

class ComplementarityReLU(NeuralNetInterface):
    """
        Builder class for creating a MPEC representation of a
        ReLU or LeakyReLU neural network on a Pyomo model.
    """
    def __init__(self,leaky_alpha = None,transform = "mpec.simple_nonlinear"):
        super().__init__()
        if leaky_alpha is not None:
            raise NotImplementedError('LeakyReLU is not yet implemented')
        self.leaky_alpha = leaky_alpha
        self.transform = transform

    def _add_activation_constraint(self,block):
        @block.Complementarity(block.INTERMEDIATE_NODES)
        def _z_hat_positive(m,i):
            return mpec.complements((m.z[i] - m.zhat[i]) >= 0, m.z[i] >= 0)
        xfrm = pyo.TransformationFactory(self.transform)
        xfrm.apply_to(block)

    def _build_neural_net(self,block):
        self._check_weights()
        build_neural_net(block,self.num_inputs,self.num_outputs,self.num_nodes,self.w,self.b)

    def build(self,block):
        self._build_neural_net(block)
        self._add_activation_constraint(block)

class TrainableNetwork(NeuralNetInterface):
    """
        Builds a Pyomo model that encodes the parameters of a neural net as variables.  The `TrainableNet` neural net builder
        can be used to train neural net parameters using Pyomo interfaced solvers.
    """
    def __init__(self,network_definition = FullSpaceNonlinear(pyo.tanh)):
        super().__init__()
        self.network_definition = network_definition

    def build(self,block):
        block.PARAMETERS = pyo.Set(initialize = list(range(len(self.w))), ordered = True)
        block.w = pyo.Var(m.PARAMETERS,initialize = self.w)
        block.b = pyo.Var(m.PARAMETERS,initialize = self.b)

        self._build_neural_net(block)
        self.network_definition._add_activation_constraint(block)

def _extract_var_data(vars):
    if isinstance(vars, ScalarVar):
        return [vars]
    elif isinstance(vars,IndexedVar):
        return list(vars.values())
    elif isinstance(vars,list):
        varlist = list()
        for v in vars:
            if v.is_indexed():
                varlist.extend(v.values())
            else:
                varlist.append(v)
        return varlist
    else:
        raise ValueError("Unknown variable type {}".format(vars))

@declare_custom_block(name='NeuralNetBlock')
class NeuralBlockData(_BlockData):

    def __init__(self, component):
        super().__init__(component)
        self._network_defn = None

    def define_network(self, *, network_definition: DefinitionInterface, input_vars=None, output_vars=None):
        self._network_defn = network_definition

        self.inputs_set = pyo.Set(initialize=range(network_definition.n_inputs()), ordered=True)
        self.outputs_set = pyo.Set(initialize=range(network_definition.n_outputs()), ordered=True)

        if input_vars is None:
            self.inputs = pyo.Var(self.inputs_set)
            self._inputs_list = list(self.inputs)
        else:
            self._inputs_list = _extract_var_data(input_vars)
            assert len(self._inputs_list) == network_definition.n_inputs()
            def _input_expr(m,i):
                return self._inputs_list[i]
            self.inputs = pyo.Expression(self.inputs_set, rule=_input_expr)

        if output_vars is None:
            self.outputs = pyo.Var(self.outputs_set)
            self._outputs_list = list(self.outputs)
        else:
            self._outputs_list = _extract_var_data(output_vars)
            assert len(self._outputs_list) == network_definition.n_outputs()
            def _output_expr(m,i):
                return self._outputs_list[i]
            self.outputs = pyo.Expression(self.outputs_set, rule=_output_expr)

        self._network_defn.build(self)