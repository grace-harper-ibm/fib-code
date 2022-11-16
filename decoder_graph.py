import pymatching as pm
import rustworkx as rx
from rustworkx.visualization import graphviz_draw


class DecoderGraph:
    """This class is probably not a mistake?"""
    
    def __init__(self, matching_graph, hori_probe_fault_id, verti_probe_fault_id, stab2node) -> None:
        self.matching_decoder = pm.Matching(matching_graph)
        self.matching_graph = matching_graph
        self.hori_probe_fault_id = hori_probe_fault_id
        self.verti_probe_fault_id = verti_probe_fault_id
        self.stab2node = stab2node
        
    def print_graph(self, fund_stab_labels=True):
        def edge_attr(edge):
            return {'label ': str(list(edge["fault_ids"])[0])}

        def node_attr(node):
            print(node)
            return {"label":str(node["element"])}
        
        if fund_stab_labels: 
            graphviz_draw(self.matching_graph, edge_attr_fn=edge_attr, node_attr_fn=node_attr)
        else:
            graphviz_draw(self.matching_graph, edge_attr_fn=edge_attr, node_attr_fn=None)
            
        

    # # this init is trash and I oppose it
    # def __init__(
    #     self,
    #     hori_graph,
    #     verti_graph,
    #     hori_prob_fault_id,
    #     verti_prob_fault_id,
    #     verti_staberr2node,
    #     verti_node2staberr,
    #     hori_staberr2node,
    #     hori_node2staberr,
    # ) -> None:
    #     self.hori_graph = hori_graph
    #     self.verti_graph = verti_graph
    #     self.hori_matching = pm.Matching(self.hori_graph)  # TODO add weights
    #     self.verti_matching = pm.Matching(self.verti_graph)
    #     self.hori_prob_fault_id = hori_prob_fault_id
    #     self.verti_prob_fault_id = verti_prob_fault_id

    #     self.hori_staberr2node = hori_staberr2node
    #     self.hori_node2staberr = hori_node2staberr

    #     self.verti_staberr2node = verti_staberr2node
    #     self.verti_node2staberr = verti_node2staberr

    def decode_prob(self, syndrome):
        """Returns whether the horizontal probe edge (aka the bottom middle bit of the triangle) was in an even (0 aka no flip) or odd (1 aka flip) number of times in the matching graph && the vertical probe"""
        res = self.matching_decoder.decode(syndrome)
        return res[self.hori_prob_fault_id], res[self.verti_prob_fault_id]