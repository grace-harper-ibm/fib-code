import pymatching as pm


class DecoderGraph():
    """This class is probably a mistake?"""
    # this init is trash and I oppose it 
    def __init__(self, hori_graph, verti_graph, hori_prob_fault_id,verti_prob_fault_id, verti_staberr2node, verti_node2staberr, hori_staberr2node, hori_node2staberr) -> None:
        self.hori_graph = hori_graph
        self.verti_graph = verti_graph
        self.hori_matching = pm.Matching(self.hori_graph) # TODO add weights 
        self.verti_matching = pm.Matching(self.verti_graph)
        self.hori_prob_fault_id = hori_prob_fault_id
        self.verti_prob_fault_id =verti_prob_fault_id 
        
        self.hori_staberr2node = hori_staberr2node
        self.hori_node2staberr = hori_node2staberr
        
        self.verti_staberr2node = verti_staberr2node
        self.verti_node2staberr = verti_node2staberr
        
        
    
    def decode_hori(self, hori_syndrome):
        """Returns whether the horizontal probe edge (aka the bottom middle bit of the triangle) was in an even (0 aka no flip) or odd (1 aka flip) number of times in the matching graph"""
        res = self.hori_matching.decode(hori_syndrome)
        return res[self.hori_prob_fault_id]

    def decode_verti(self, verti_syndrome):        
        """Returns whether the vertical probe edge (aka if you look at the row where the bottom of the triangle is, there will be 1 bit that isn't on/in the triangle. this is that bit) was in an even (0 aka no flip) or odd (1 aka flip) number of times in the matching graph"""
        res = self.hori_matching.decode(verti_syndrome)
        return res[self.verti_prob_fault_id]
        
        
        
        
        