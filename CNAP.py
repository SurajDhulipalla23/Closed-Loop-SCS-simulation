from neuron import h, gui
from neuron.units import ms, mV
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import pandas as pd
# put a point source at the recording electrode, calculate the voltage at each node, and sum and multiple by current to get the recording electrode voltage value
h.load_file('stdrun.hoc')

h.tstop = 10 # [ms]: simulation
h.dt = 0.01 # [ms]: timestep

np.random.seed(0)
def generate_axon_diameters(num_axons, mean_diameter, std_diameter):
    diameters = np.random.normal(loc=mean_diameter, scale=std_diameter, size=num_axons)
    return diameters

# Generate axon diameters
n_axons = 100
mean_diameter_small = 9  # µm
std_dev = 2
axon_diameters = generate_axon_diameters(n_axons, mean_diameter_small, std_dev)
axon_diameters = np.round(axon_diameters, 2)

potential = []
# Define axons and set positions
axon_list = []
for axon_index, D in enumerate(axon_diameters):

    Vo = -80 # [mV]: resting membrane potential
    n_nodes = 81 #  number of sections
    #D = 11 #[um]
    inl = 100*D # [um]: internodal length
    rhoa = 54.7 # [ohm*cm]: axoplasmic resistivity
    cm = 2.5 # [uF/cm**2]: specific membrane capacitance
    L = 1.5 # [um]
    nseg = 1 # spatial res

    # material parameters
    sigma_e = 1/5000 # [S/mm]: extracellular medium conductivity

    #Simulation parameters
    #0.1 ms (0.2 previously) monophasic cathodic pulse for Problem 2.3
    delay = 1 # [ms]
    amp = 6 # [mA]
    e2f = 3.5 # [mm]
    dur = 0.1 # [ms]
    
    nodes = [h.Section(name=f'axon[{axon_index}].node[{i}]') for i in range(n_nodes)]
    node_x = []
    node_y = []
    node_z = []
    for node_ind, node in enumerate(nodes):
        node.nseg = nseg
        node.diam = 0.6 * D
        node.L = L
        node.Ra = rhoa * ((L + inl) / L)
        node.cm = cm
        node.insert('extracellular')
        for seg in node:
            seg.extracellular.e = 0
        if node_ind > 0:
            node.connect(nodes[node_ind-1](1))

    # print("X positions: {}".format(node_x)) # X position not working in gaussian
    # print("Y positions: {}".format(node_y)) # Y position not working in gaussian
    # print("Z positions: {}".format(node_z))
    

    axon_list.append(nodes)

    # INSTRUMENTATION - STIMULATION/RECORDING
    dummy = h.Section(name='dummy')
    # dummy.insert('extracellular')
    dummy = h.Section(name='dummy')
    e_stim_obj = h.IClamp(dummy(0.5)) # puts the stim halfway along the length
    e_stim_obj.delay = delay
    e_stim_obj.dur = dur
    e_stim_obj.amp = amp 

    vol_mem = [h.Vector().record(sec(0.5)._ref_v) for sec in nodes]
    tvec = h.Vector().record(h._ref_t)
    im_rec = [h.Vector().record(sec(0.5)._ref_i_membrane) for sec in nodes]

    # Determine the location of each node for interpolation
    # print("node positions = {node_positions}".format(node_positions = node_positions))
    # Interpolate electric field data to find potential at each node

    def update_field():
        phi_e = []
        for node_ind, node in enumerate(nodes):
            x_loc = 1e-3*(-(n_nodes-1)/2*inl + inl*node_ind) # 1e-3 [um] -> [mm]
            r = np.sqrt(x_loc**2 + e2f**2) # [mm]

            phi_e.append(e_stim_obj.i/(4*sigma_e*np.pi*r))
            node(0.5).e_extracellular = phi_e[node_ind]
    
    t_array = []

    # time integrate with constant time step
    def my_advance():
        update_field()
        global voltage
        t = len(t_array) * h.dt
        t_array.append(t)
        h.fadvance()
    # this is somewhat of a "hack" to change the default run procedure in HOC
    h(r"""
    proc advance() {
    nrnpython("my_advance()")
    }""")

    h.finitialize(Vo)
    h.continuerun(h.tstop)
    vol_sum = np.sum(vol_mem, axis=0)

    potential.append(vol_sum)
# print(potential)
large_total_voltage = []

large_total_voltage = np.sum(potential, axis=0)
total_voltage = np.sum(potential, axis = 0)
# potential_list.append(total_voltage)


plt.plot(tvec, large_total_voltage, label = "total voltage")
# plt.plot(t_array, small_total_voltage, label = "small diameter fibers")
# plt.plot(t_array, potential_list, label = "large diameter fibers")
plt.xlabel("Time (ms)")
plt.ylabel("Transmembrane Potential (mV)")
plt.title("Compound Nerve Action Potential (CNAP)")
# plt.legend(["total_voltage", "small diameter fibers", "large diameter fibers"], loc="lower right")
plt.show()

