#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UC Irvine CS274B
"""

import numpy as np
from copy import deepcopy
from pgmpy.models import FactorGraph
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.factors import factor_product
from pgmpy.readwrite import BIFReader
from pgmpy.inference.ExactInference import BeliefPropagation


class SimpleFactorGraph(FactorGraph):
    # simplify some methods and error-checking in FactorGraph for efficiency
    def __init__(self):
        super(SimpleFactorGraph, self).__init__()
        self.card_dict = {}

    def get_variable_nodes(self):
        variable_nodes = set([x for factor in self.factors for x in factor.scope()])
        return list(variable_nodes)

    def add_factors(self, *factors):
        for factor in factors:
            self.factors.append(factor)
            for var in factor.scope():
                self.card_dict[var] = factor.get_cardinality([var])[var]

    def remove_node(self, f):
        super().remove_node(f)
        if f in self.card_dict.keys():
            del self.card_dict[f]

    def get_cardinality(self, variable_name=None):
        """
        Returns the cardinality of the node or a dict over all nodes if not specified.
        """
        if variable_name is not None:
            return self.card_dict[variable_name]
        else:
            return self.card_dict

    def add_evidence(self, var, val):
        """
        ADD_EVIDENCE - Adds the "evidence" that variable "var" takes
        on value "val".  This slices the factors neighboring "var" accordingly
        and returns the updated factor graph structure.

        Parameters
        ----------
        G : Factor Graph
        var : variable node of the factor graph
        val : value of variable evidence

        Returns
        -------
        G : modified factor graph involving evidence information
        """
        var_dim = self.get_cardinality(var)
        # iterate factor neighbors
        fac_nbrs = deepcopy(list(self.neighbors(var)))

        for fac_nbr in fac_nbrs:
            potential = fac_nbr.values
            var_nbrs = fac_nbr.scope()
            card = fac_nbr.cardinality

            self.remove_factors(fac_nbr)

            if len(var_nbrs) == 1:
                continue
                # new_var_nbrs = var_nbrs
                # new_card = card
                # new_potential = np.zeros(card[0])
                # new_potential[val] = 1
            else:
                I = var_nbrs.index(var)
                ind = [slice(None)] * len(var_nbrs)
                ind[I] = val
                new_potential = potential[tuple(ind)]
                var_nbrs.remove(var)
                new_var_nbrs = var_nbrs
                l_card = list(card)
                l_card.pop(I)
                new_card = np.array(l_card)
            phi = FastDiscreteFactor(new_var_nbrs, new_card, new_potential)
            if phi in self.factors:
                continue

            self.add_factors(phi)
            self.add_nodes_from([phi])
            edges = []
            for variable in new_var_nbrs:
                edges.append((variable, phi))
            self.add_edges_from(edges)

        # marginal = np.zeros(var_dim)
        # marginal[val] = 1
        # phi = FastDiscreteFactor([var], [var_dim], marginal)
        # self.add_factors(phi)
        # self.add_nodes_from([phi])
        # self.add_edges_from([(var, phi)])
        self.remove_node(var)

        return self

    def add_evidences(self, vars, vals):
        for var, val in zip(vars, vals):
            self.add_evidence(var, val)


class FastDiscreteFactor(DiscreteFactor):
    # define faster hash function, depending only on variable names
    def __hash__(self):
        variable_hashes = [hash(variable) for variable in self.variables]
        return hash(sum(variable_hashes))
        # return hash(str(sorted(variable_hashes)))


def make_debug_graph():
    G = SimpleFactorGraph()
    G.add_nodes_from(['x1', 'x2', 'x3', 'x4'])

    # add factors 
    phi1 = FastDiscreteFactor(['x1', 'x2'], [2, 3], np.array([0.5, 0.7, 0.2,
                                                              0.5, 0.3, 0.8]))
    phi2 = FastDiscreteFactor(['x2', 'x3', 'x4'], [3, 2, 2], np.array([0.2, 0.25, 0.70, 0.30,
                                                                       0.4, 0.25, 0.15, 0.65,
                                                                       0.4, 0.50, 0.15, 0.05]))
    phi3 = FastDiscreteFactor(['x3'], [2], np.array([0.5, 0.5]))
    phi4 = FastDiscreteFactor(['x4'], [2], np.array([0.4, 0.6]))
    G.add_factors(phi1, phi2, phi3, phi4)
    G.add_nodes_from([phi1, phi2, phi3, phi4])
    G.add_edges_from([('x1', phi1), ('x2', phi1), ('x2', phi2), ('x3', phi2), ('x4', phi2), ('x3', phi3), ('x4', phi4)])

    return G


def make_alarm_graph():
    """
    make a factor graph from the Bayesian model of the alarm network

    Returns
    -------
    G : factor graph.

    """
    reader = BIFReader('alarm.bif')

    alarm_model = reader.get_model()

    G = SimpleFactorGraph()
    G.add_nodes_from(alarm_model.nodes())

    # make factors and edges
    for cpd in alarm_model.get_cpds():

        variables = cpd.variables
        card = cpd.cardinality
        values = cpd.values

        phi = FastDiscreteFactor(variables, card, values)
        G.add_factors(phi)
        G.add_nodes_from([phi])
        edges = []
        for variable in variables:
            edges.append((variable, phi))
        G.add_edges_from(edges)

    return G


def make_alarm_graph_partA():
    """
    make a small factor graph from part of the Bayesian model of the alarm network

    Returns
    -------
    G : factor graph.

    """
    reader = BIFReader('alarm.bif')

    alarm_model = reader.get_model()

    G = SimpleFactorGraph()
    variable_nodes = ['PULMEMBOLUS', 'PAP', 'INTUBATION', 'SHUNT', 'VENTTUBE', 'KINKEDTUBE', 'PRESS', 'VENTLUNG']
    G.add_nodes_from(variable_nodes)
    for variable_node in variable_nodes:
        if variable_node == 'VENTTUBE':
            # Define prior conditioning on VENTMACH=3, DISCONNECT=1
            phi = FastDiscreteFactor(['VENTTUBE'], [4], np.array([0.01, 0.01, 0.01, 0.97]))
            G.add_factors(phi)
            G.add_nodes_from([phi])
            G.add_edges_from([('VENTTUBE', phi)])
        else:
            cpd = alarm_model.get_cpds(variable_node)
            variables = cpd.variables
            card = cpd.cardinality
            values = cpd.values

            phi = FastDiscreteFactor(variables, card, values)
            G.add_factors(phi)
            G.add_nodes_from([phi])
            edges = []
            for variable in variables:
                edges.append((variable, phi))
            G.add_edges_from(edges)

    return G


def make_alarm_graph_partC():
    """
    make a small factor graph from part of the Bayesian model of the alarm network

    Returns
    -------
    G : factor graph.
    """
    reader = BIFReader('alarm.bif')

    alarm_model = reader.get_model()

    G = SimpleFactorGraph()
    variable_nodes = ['INTUBATION', 'VENTTUBE', 'KINKEDTUBE', 'VENTLUNG', 'PRESS', 'MINVOL']
    G.add_nodes_from(variable_nodes)
    for variable_node in variable_nodes:
        if variable_node == 'VENTTUBE':
            # Define prior conditioning on VENTMACH=3, DISCONNECT=1
            phi = FastDiscreteFactor(['VENTTUBE'], [4], np.array([0.01, 0.01, 0.01, 0.97]))
            G.add_factors(phi)
            G.add_nodes_from([phi])
            G.add_edges_from([('VENTTUBE', phi)])
        else:
            cpd = alarm_model.get_cpds(variable_node)
            variables = cpd.variables
            card = cpd.cardinality
            values = cpd.values

            phi = FastDiscreteFactor(variables, card, values)
            G.add_factors(phi)
            G.add_nodes_from([phi])
            edges = []
            for variable in variables:
                edges.append((variable, phi))
            G.add_edges_from(edges)

    G.add_evidence('MINVOL', 1)
    G.add_evidence('PRESS', 3)

    return G


def marg_brute_force(G):
    """
    MARG_BRUTE_FORCE - Compute marginals by brute force enumeration.
    This is mainly used for debugging on smaller graphical models.

    Parameters
    ----------
    G : Factor Graph

    Returns
    -------
    nodeMarg : dictionary of marginal distribution of each variable
    """

    nodeMarg = {}
    num_vars = len(G.get_variable_nodes())
    # compute joint distribution
    factors = G.get_factors()
    factor = factors[0]
    factor = factor_product(factor, *[factors[i] for i in range(1, len(factors))])
    joint = factor.values

    for i in range(num_vars):
        I = [j for j in range(num_vars)]
        I.remove(i)
        thisMarg = np.sum(joint, tuple(I))
        thisMarg = thisMarg.reshape(len(thisMarg), 1)  # keep dimension consistent with beliefs
        thisMarg = thisMarg / np.sum(thisMarg)
        nodeMarg[factor.scope()[i]] = thisMarg

    return nodeMarg


def marg_junction_tree(G, eval=None, vars=None, vals=None):
    """
    MARG_JUNCTION_TREE - Compute marginals by applying belief propagation to the junction tree.
    This can be used to compute the ground truth for graphical models of medium size.

    Parameters
    ----------
    G : Factor Graph without evidence
    vars : list of variable nodes of the factor graph for which we have evidence
    vals : list of corresponding values of variable evidence
    eval : list of variable whose expected values to output, defaults to all

    Returns
    -------
    nodeMarg : dictionary of marginal distribution of each variable
    """

    if eval is None:
        eval = G.get_variable_nodes()
    bp = BeliefPropagation(G)
    if vars is not None and vals is not None:
        nodeMarg = bp.query(eval, {k: v for k, v in zip(vars, vals)}, joint=False)
    else:
        nodeMarg = bp.query(eval, joint=False)
    nodeMarg = {k: v.values.reshape(-1, 1) for k, v in nodeMarg.items()}
    return nodeMarg


def init_message(G):
    """
    INIT_MESSAGES - Initialize factor-to-variable node messages to empty.

    Returns
    -------
    M : dictionary of initialized factor-to-variable message
    """

    # init msg dictionary
    M = {}
    for var_j in G.get_variable_nodes():
        M[var_j] = {}

    # init empty messages
    for fac_i in G.get_factors():
        for var_j in fac_i.scope():
            M[var_j][fac_i] = np.ones([fac_i.get_cardinality([var_j])[var_j], 1])
    return M


def generate_schedule(G):
    """
    GENERATE_SCHEDULE - Creates a message update schedule for the factor graph 'G'

    Returns
    -------
    sched : list of message update schedule
    """

    sched = []
    for fac in G.get_factors():
        for var in fac.scope():
            sched.append([fac, var])

    return sched


def update_fac2var_msg(G, fac, var, M_in):
    """
    UPDATE_FAC2VAR_MSG - Updates the factor-to-variable node message.

    Parameters
    ----------
    G : Factor Graph
    fac : factor
    var : variable node
    M_in : dictionary of incoming messages

    Returns
    -------
    msg: numpy array of
         normalized factor-to-variable message

    """

    fac_to_var = fac.scope()
    nVars = len(fac_to_var)

    # compute product over all the other messages
    sum_message = 0
    for key in M_in.keys():
        message = np.log(np.maximum(M_in[key], np.finfo(float).tiny))
        dim = fac_to_var.index(key)
        sz = np.ones(nVars)
        sz[dim] = len(message)
        message = np.reshape(message, sz.astype(int))
        sum_message = sum_message + message

    # multiply product of messages with potential function
    max_message = np.max(sum_message)
    sum_message = sum_message - max_message
    totalSum = fac.values * np.exp(sum_message)

    # sum over all other variables
    for key in M_in.keys():
        dim = fac_to_var.index(key)
        totalSum = np.sum(totalSum, dim, keepdims=True)

    msg = np.reshape(totalSum, [np.size(totalSum), 1]) / np.sum(totalSum)

    return msg


def get_msg_var2fac(G, var, fac, M):
    """
    GET_MSG_VAR2FAC - Returns variable-to-factor messages.

    Parameters
    ----------
    G :Factor Graph.
    var : variable nodes
    fac : factor

    Returns
    -------
    M_out: dictionary containing messages


    """

    M_out = {}

    for vi in var:
        MM = np.ones([G.get_cardinality(vi), 1])

        nbrs_fac = list(G.neighbors(vi))
        nbrs_fac.remove(fac)
        for nbr in nbrs_fac:
            MM *= M[vi][nbr]

        M_out[vi] = MM / np.sum(MM)

    return M_out


def get_beliefs(G, M, variable_nodes=[]):
    """
    GET_BELIEFS - Returns dictionary containing beliefs for each node and
    each clique, respectively.

    Parameters
    ----------
    G : Factor Graph.
    M : factor-to-variable message dictionary
    variable_nodes : optional, precomputed G.get_variable_nodes()

    Returns
    -------
    node_marg : dictionary containing beliefs for each node
                same format as nodeMarg = marg_brute_force(G)

    """

    # variable node names
    if len(variable_nodes) == 0:
        variable_nodes = G.get_variable_nodes()

    # node beliefs
    node_marg = {}

    for vi in variable_nodes:
        MM = 1
        for msg in M[vi].values():
            MM *= msg

        node_marg[vi] = MM / np.sum(MM)

    return node_marg


def check_converged(M_new, M_old, thresh):
    """
    CHECK_CONVERGED - Checks message convergence.  Messages are "converged"
    iff the maximum absolute difference between M_new and M_old is below
    'thresh'.

    Parameters
    ----------
    M_new : new messages
    M_old : old messages
    thresh : convergence tolerance

    Returns
    -------
    converged : boolean variable indicates whether the message converges

    """

    max_dist = 0
    for vv in M_new.keys():
        for (ff, msg) in M_new[vv].items():
            this_dist = np.sqrt(np.sum(np.square(msg - M_old[vv][ff])))
            max_dist = np.maximum(max_dist, this_dist)

    converged = max_dist <= thresh

    return converged


def run_loopy_bp_parallel(G, max_iters, conv_tol):
    """
    RUN_LOOPY_BP - Runs Loopy Belief Propagation (Sum-Product) on a factor
    Graph given by 'G'.This implements a "parallel" updating scheme in
    which all factor-to-variable messages are updated in a single clock
    cycle, and then all variable-to-factor messages are updated.

    Parameters
    ----------
    G : Factor Graph
    max_iters : max iterations
    conv_tol : convergence tolerance

    Returns
    -------
    nodeMargs : list keeping track of node marginals at each iteration.
                NodeMargs[i] is a dictionary containing beliefs for each node 
                with the same format as nodeMarg = marg_brute_force(G)
    """

    variable_nodes = G.get_variable_nodes()

    M = init_message(G)
    sched = generate_schedule(G)
    nodeMargs = []
    num_msgs = len(sched)

    for iters in range(max_iters):
        M_old = {var: fac.copy() for var, fac in M.items()}
        # M_old = deepcopy(M)

        for I in range(num_msgs):
            fac_i = sched[I][0]
            var_j = sched[I][1]

            # get incoming messages
            nbrs_var = list(G.neighbors(fac_i))
            nbrs_var.remove(var_j)
            preM = get_msg_var2fac(G, nbrs_var, fac_i, M_old)
            msg = update_fac2var_msg(G, fac_i, var_j, preM)
            M[var_j][fac_i] = msg

        # keep track of node marginals at each iteration    
        nodeMargs.append(get_beliefs(G, M, variable_nodes))

        # check message convergence
        if iters > 0 and check_converged(M, M_old, conv_tol):
            print('Parallel LBP converged in', iters, 'iterations.')
            break
    else:
        print('Parallel LBP did not converge. Terminated at', max_iters, 'iterations')

    return nodeMargs


def run_loopy_bp_parallel2(G, max_iters, conv_tol):
    """
    RUN_LOOPY_BP - Runs Loopy Belief Propagation (Sum-Product) on a factor
    Graph given by 'G'.This implements a "parallel" updating scheme in
    which all factor-to-variable messages are updated in a single clock
    cycle, and then all variable-to-factor messages are updated.

    Parameters
    ----------
    G : Factor Graph
    max_iters : max iterations
    conv_tol : convergence tolerance

    Returns
    -------
    nodeMargs : list keeping track of node marginals at each iteration.
                NodeMargs[i] is a dictionary containing beliefs for each node 
                with the same format as nodeMarg = marg_brute_force(G)
    """

    variable_nodes = G.get_variable_nodes()

    M = init_message(G)
    sched = generate_schedule(G)
    num_msgs = len(sched)

    nodeMarg = None
    for iters in range(max_iters):
        M_old = {var: fac.copy() for var, fac in M.items()}
        # M_old = deepcopy(M)

        for I in range(num_msgs):
            fac_i = sched[I][0]
            var_j = sched[I][1]

            # get incoming messages
            nbrs_var = list(G.neighbors(fac_i))
            nbrs_var.remove(var_j)
            preM = get_msg_var2fac(G, nbrs_var, fac_i, M_old)
            msg = update_fac2var_msg(G, fac_i, var_j, preM)
            M[var_j][fac_i] = msg

        # keep track of node marginals at each iteration    
        del nodeMarg
        nodeMarg = get_beliefs(G, M, variable_nodes)

        # check message convergence
        if iters > 0 and check_converged(M, M_old, conv_tol):
            print('Parallel LBP converged in', iters, 'iterations.')
            break
    else:
        print('Parallel LBP did not converge. Terminated at', max_iters, 'iterations')

    return nodeMarg



def belief_diff(b1, b2):
    """
    BELIEF_DIFF - Computes the symmetric L1 distance between belief's 'b1'
    and 'b2'. Inputs are N-entry dictionaries, where each dictionary stores
    the marginal distribution of each variable.  L1 distances are returned
    in an N-dim numpy array.
    """

    num_b = len(b1)
    D = np.zeros(num_b)
    for i, key in enumerate(b1.keys()):
        D[i] = 0.5 * np.sum(abs(b1[key] - b2[key]))

    return D
