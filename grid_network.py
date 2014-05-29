import collections
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

GridGraph = collections.namedtuple('GridGraph', ['graph', 'width', 'height', 'periodic'])

def build_grid(w, h, periodic=False):
    g = nx.grid_2d_graph(w, h, periodic=periodic)
    for i, j, data in g.edges(data=True):
        data['weight'] = 1
    return GridGraph(g, w, h, periodic)

def nodes(g):
    node_order = []
    for y in range(g.height):
        for x in range(g.width):
            node_order.append((x,y))
    return node_order

def node_vector(g, x, y):
    v = np.zeros((g.width * g.height))
    v[y * g.height + x] = 1
    return v

def laplacian(g):
    node_order = nodes(g)
    return np.asarray(nx.laplacian_matrix(g.graph, node_order))

def effective_resistance(g, (x1, y1), (x2, y2)):
    v = node_vector(g, x1, y1) - node_vector(g, x2, y2)
    tmp = np.dot(np.linalg.pinv(laplacian(g)), v)
    return np.dot(v, tmp)

def set_resistance(g, (x1, y1), (x2, y2), resistance):
    g.graph[(x1,y1)][(x2,y2)]['weight'] = 1.0 / resistance

def run_experiment(w, h, u, v, e1, e2, periodic=False):
    g = build_grid(w, h, periodic=periodic)
    g.graph.remove_edge(e1, e2)
    return effective_resistance(g, u, v)

def test_all_edges(w, h, u, v, periodic=False):
    g = build_grid(w, h, periodic=periodic)
    baseline_resistance = effective_resistance(g, u, v)
    result = []
    all_edges = list(g.graph.edges())
    for i, j in all_edges:
        g.graph.remove_edge(i, j)
        resistance = effective_resistance(g, u, v)
        g.graph.add_edge(i, j)
        g.graph[i][j]['weight'] = 1
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