class SimpleHardware:
    time_unit = "us"

    def __init__(self):
        super().__init__()
        self.two_qubit_gate_error = 0
        self.one_qubit_gate_error = 0
        self.readout_error = 0
        self.two_qubit_gate_time = 0
        self.one_qubit_gate_time = 0
        self.readout_time = 0
        self.init_time = 0
        self.connectivity = []
        self.n_qpu = 1
        self.n_qubit = -1

    def set_errors(self, one_qubit=0.0, two_qubit=0.0, readout=0.0):
        self.one_qubit_gate_error = one_qubit
        self.two_qubit_gate_error = two_qubit
        self.readout_error = readout

    def set_times(self, one_qubit=0.0, two_qubit=0.0, readout=0.0, initialization=0.0):
        self.two_qubit_gate_time = two_qubit
        self.one_qubit_gate_time = one_qubit
        self.readout_time = readout
        self.init_time = initialization

    def set_extra(self, n_qubit=None, connectivity=None, n_qpu=None):
        self.n_qubit = n_qubit
        self.n_qpu = n_qpu
        self.connectivity = connectivity