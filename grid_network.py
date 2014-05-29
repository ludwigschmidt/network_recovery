import collections
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

GridGraph = collections.namedtuple('GridGraph', ['graph', 'width', 'height', 'grid_type'])

ground_node = (-1, -1)
no_resistance = 1e10
normal_resistance = 1
large_resistance = 1e-10

def build_grid(w, h, grid_type):
    if grid_type == 'plain' or grid_type == 'periodic':
        g = nx.grid_2d_graph(w, h, periodic=(grid_type == 'periodic'))
        for i, j, data in g.edges(data=True):
            data['weight'] = normal_resistance
        return GridGraph(g, w, h, grid_type)
    if grid_type == 'grounded_boundary':
        g = nx.grid_2d_graph(w, h, periodic=False)
        for i, j, data in g.edges(data=True):
            data['weight'] = normal_resistance
        g.add_node(ground_node)
        for xx in range(w):
            g.add_edge(ground_node, (xx, 0), weight=no_resistance)
            g.add_edge(ground_node, (xx, h - 1), weight=no_resistance)
        for yy in range(h):
            g.add_edge(ground_node, (0, yy), weight=no_resistance)
            g.add_edge(ground_node, (w - 1, yy), weight=no_resistance)
        return GridGraph(g, w, h, grid_type)
    print 'Unknown grid_type!'
    return []


def nodes(g):
    node_order = []
    for y in range(g.height):
        for x in range(g.width):
            node_order.append((x,y))
    if g.grid_type == 'grounded_boundary':
        node_order.append(ground_node)
    return node_order


def node_vector(g, node):
    if g.grid_type == 'grounded_boundary':
        size = g.width * g.height + 1
    else:
        size = g.width * g.height
    v = np.zeros((size))
    if node != ground_node:
        v[node[1] * g.height + node[0]] = 1
    else:
        v[size - 1] = 1
    return v


def laplacian(g):
    node_order = nodes(g)
    return np.asarray(nx.laplacian_matrix(g.graph, node_order))


def effective_resistance(g, node1, node2):
    v = node_vector(g, node1) - node_vector(g, node2)
    tmp = np.dot(np.linalg.pinv(laplacian(g)), v)
    return np.dot(v, tmp)
    

def get_grounded_potentials(g, probe_node):
    if g.grid_type != 'grounded_boundary':
        print 'Error: wrong grid type!'
        return []
    v = node_vector(g, probe_node) - node_vector(g, ground_node)
    tmp = np.linalg.pinv(laplacian(g))
    vec_res = np.dot(tmp, v)
    node_order = nodes(g)
    res = {}
    ground_potential = vec_res[-1]
    for ii, node in enumerate(node_order):
        res[node] = vec_res[ii] - ground_potential
    return res
    

def set_resistance(g, node1, node2, resistance):
    g.graph[node1][node2]['weight'] = 1.0 / resistance


def run_experiment(w, h, u, v, e1, e2, periodic=False):
    g = build_grid(w, h, periodic=periodic)
    g.graph.remove_edge(e1, e2)
    return effective_resistance(g, u, v)


# test all edges in the following setup:
# grid of size w, h (maybe periodic)
# for each edge: remove the edge and measure the change in effective resistance between u and v
def test_all_edges(w, h, u, v, periodic=False):
    g = build_grid(w, h, periodic=periodic)
    baseline_resistance = effective_resistance(g, u, v)
    result = []
    all_edges = list(g.graph.edges())
    for i, j in all_edges:
        g.graph.remove_edge(i, j)
        resistance = effective_resistance(g, u, v)
        g.graph.add_edge(i, j)
        g.graph[i][j]['weight'] = normal_resistance
        result.append((i, j, resistance - baseline_resistance))
    return result, g


def test_all_edges_and_compute_dsts(w, h, u, v, periodic=False):
    (result, g) = test_all_edges(w, h, u, v, periodic=periodic)
    tmp1 = [(dst(u, e1, periodic=periodic, w=w, h=h), dst(u, e2, periodic=periodic, w=w, h=h),
             dst(v, e1, periodic=periodic, w=w, h=h), dst(v, e2, periodic=periodic, w=w, h=h), (e1, e2), res) for (e1, e2, res) in result]
    tmp2 = [((e1, e2), min(d1, d2, d3, d4), res) for (d1, d2, d3, d4, (e1, e2), res) in tmp1]
    return tmp2, g


def test_all_edges_and_plot(w, h, u, v, periodic=False):
    result, g = test_all_edges_and_compute_dsts(w, h, u, v, periodic=periodic)
    plotvals = zip(*result)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_yscale('log')
    ax.set_xlim(min(plotvals[1]) - .5, max(plotvals[1]) + .5)
    y_bounded = map(lambda x: x if x > 1e-16 else 1e-16, plotvals[2])
    ax.scatter(plotvals[1], y_bounded)
    ax.set_ylabel('Change in effective resistance')
    ax.set_xlabel('Distance from edge to test nodes')
    title = 'Test nodes: ({},{}) and ({},{})'.format(u[0], u[1], v[0], v[1])
    if periodic:
        title += '   (periodic grid)'
    ax.set_title(title)
    return fig


def test_all_edges_and_save_plot(w, h, u, v, periodic=False):
    filename = 'test_'
    if periodic:
        filename += 'periodic_'
    filename += '{}_{}__{}_{}__{}_{}.pdf'.format(w, h, u[0], u[1], v[0], v[1])
    fig = test_all_edges_and_plot(w, h, u, v, periodic)
    fig.savefig(filename, bbox_inches='tight')
    return fig


def dst((x1, y1), (x2, y2), periodic=False, w=-1, h=-1):
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1
    if periodic:
        minx = min(x2 - x1, x1 + w - x2)
        miny = min(y2 - y1, y1 + h - y2)
        return minx + miny
    else:
        return x2 - x1 + y2 - y1