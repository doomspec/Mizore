from mizore.comp_graph.comp_node import CompNode


class CalcNode(CompNode):

    def __init__(self, name=None):
        super().__init__(name=name)

    def calc(self, cache_key=None):
        pass

    def __call__(self, *args, **kwargs):
        return self.outputs
