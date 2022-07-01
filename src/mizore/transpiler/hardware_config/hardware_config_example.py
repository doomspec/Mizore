from mizore.transpiler.hardware_config.simple_hardware import SimpleHardware


def example_superconducting_circuit_config():
    # Unit: us
    config = SimpleHardware()
    config.set_errors(one_qubit=0.00001, two_qubit=0.0001, readout=0.003)
    config.set_times(one_qubit=0.02, two_qubit=0.2, readout=0.1, initialization=0.1)
    return config
