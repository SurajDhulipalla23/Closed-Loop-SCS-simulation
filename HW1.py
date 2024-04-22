# import NEURON library
from neuron import h

# other imports
import matplotlib.pyplot as plt
import numpy as np
from tqdm.notebook import tqdm
from unyt import mA, uA, nA, cm, mm
from scipy.interpolate import griddata
import pandas as pd

# Threshold search
# Only the current currently included (no PW or frequency)
# Make the fiber in the z direction
# Have to have a ECAP (signal being used to close the loop), the number of fibers activated = ECAP response
# i_membrane multiply surface area of node (nodal current)
# Vr = SUM(I/4*pi*sigma*r) (I = membrane currents)
# Fully exposed cylindrical lead

file_name = 'ElectricalPotential.csv'

electric_field_data = np.genfromtxt(file_name, delimiter=',', skip_header=9)
df = pd.DataFrame(electric_field_data, columns=['x-value', 'y-value', 'E-field'])
df['x-value'] *= 1000
df['y-value'] *= 1000

# Assuming E-field is directly proportional to the distance from the origin and its direction
# Calculate vector components
df['Ex'] = df['E-field'] * df['x-value'] / np.sqrt(df['x-value']**2 + df['y-value']**2)
df['Ey'] = df['E-field'] * df['y-value'] / np.sqrt(df['x-value']**2 + df['y-value']**2)

# Handle cases where both x and y are 0 to avoid division by zero
df.fillna(0, inplace=True)

# Plotting
fig, ax = plt.subplots()
q = ax.quiver(df['x-value'], df['y-value'], df['Ex'], df['Ey'], df['E-field'], scale=5)
ax.quiverkey(q, X=0.3, Y=1.1, U=10, label='Electric Field Magnitude (units)', labelpos='E')

plt.xlabel('X position (mm)')
plt.ylabel('Y position (mm)')
plt.title('Electric Field Visualization')
plt.show()


# MODEL SPECIFICATION
tstop = 5 # [ms]: simulation time
h.dt = 0.001 # [ms]: timestep
n_nodes = 51
time_steps = int(tstop / h.dt) + 1  # +1 to include the endpoint

def run(D, dist, electric_field_data, threshold=20):
    """
    D: fiber diameter, um
    dist: r (electrode-fiber distance), cm
    amp: stimulus amplitude, mA (default: -1 mA)
    pw: pulse width, ms (default: 0.1 ms)
    threshold: threshold voltage, mV (default: +20mV)
    """
    dist = float(dist.to('cm'))

    # cell params
    v_init = 0 # [mV]: Vm @ rest, initial condition
    n_nodes = 51 # []: (int) number of sections, make this an odd number
    inl = 100 * D # [um]: internodal length
    rhoa = 100 # [ohm-cm]: axoplasmic/axial resistivity
    cm = 1 # [uF/cm**2]
    L = 1.5 # [um]
    nseg = 1 # []: (int)
    g = 1/2000 # [S/cm**2]
    # material params
    sigma_e = 2e-3 # [S/cm]: extracellular medium resistivity
    # stim params
    delay = 0.5 # [ms]: start time of stim

    # CONSTRUCT FIBER
    nodes = [h.Section(name=f'node[{i}]') for i in range(n_nodes)]

    for node_ind, node in enumerate(nodes):
        node.nseg = nseg
        node.diam = 0.7 * D
        node.L = L
        node.Ra = rhoa*((L+inl)/L)
        node.cm = cm
        node.insert('pas')
        node.insert('extracellular')
        # node.insert('hh')

        for seg in node:
            seg.pas.g = g
            seg.pas.e = v_init
            seg.extracellular.e = 0

        if node_ind > 0:
            node.connect(nodes[node_ind-1](1))

    # INSTRUMENTATION - STIMULATION/RECORDING
    # make dummy object to "host" the stim
    # dummy = h.Section(name='dummy')

    # # construct stimulus
    # e_stim_obj = h.IClamp(dummy(0.5))
    # e_stim_obj.delay = delay
    # e_stim_obj.dur = pw
    # e_stim_obj.amp = amp

    # # calculate extracellular potentials
    # phi_e = []
    # start = -((n_nodes-1)/2)*(inl+L)

    # for node_ind, node in enumerate(nodes):
    #     x_loc = 1e-4*(start + (inl+L)*node_ind) # 1e-4 [um] -> [cm]
    #     r = np.sqrt(x_loc**2 + dist**2) # [cm]
    #     phi_e.append(1/(4*sigma_e*np.pi*r)) 

    # h.t = 0
    # neuron vectors for recording membrane potentials
    vol_mem = [h.Vector().record(sec(0.5)._ref_v) for sec in nodes]
    # i_nas = [h.Vector().record(sec(0.5)._ref_ina) for sec in nodes]
    # i_membrane = [h.Vector().record(sec(0.5)._ref_i_membrane_) for sec in nodes]
    tvec = h.Vector().record(h._ref_t)

    # RUN MODEL
    # initialize
    h.finitialize(v_init)


    # thresh = (np.abs(amp)*mA) * (threshold / (np.max(vol_mem) - v_init))
    time_steps = int(tstop / h.dt)
    
    # Determine the location of each node for interpolation
    node_positions = np.linspace(-n_nodes//2, n_nodes//2, n_nodes) * (1e-5*(L + inl))
    print("node positions = {node_positions}".format(node_positions = node_positions))
    # Interpolate electric field data to find potential at each node
    points = electric_field_data[:, :2]  # x, y positions
    values = electric_field_data[:, 2]   # Electric potential
    node_potentials = griddata(points, values, (node_positions, np.zeros_like(node_positions)), method='linear')
    print(node_potentials)

    # Simulation loop
    while h.t < tstop:
        if h.t >= 0.5 and h.t <= 0.8:
            for node, phi in zip(nodes, node_potentials):
                node(0.5).e_extracellular = phi
        else: 
            for node, phi in zip(nodes, node_potentials):
                node(0.5).e_extracellular = 0
        h.fadvance()
    all_membrane_potentials = np.array([np.array(vm) for vm in vol_mem])
    # i_membrane = []
    # for seg in nodes:
    #     i_m = h.Vector().record(seg._ref_i_membrane_)  # Total membrane current
    # #     i_membrane.append(i_m)
    # all_currents = np.array([np.array(i) for i in i_membrane])

    return all_membrane_potentials, np.array(tvec), 0

D = 10  # Example diameter
dist = 1*cm  # Example distance
threshold = 20  # Example threshold

electric_field_data2 = df.to_numpy()
# Run the simulation with the example electric field data
membrane_potentials, time_vector, transmembrane_current = run(D, dist, electric_field_data2, threshold)

# Plotting the membrane potential of the 26th segment over time
for i, membrane_potential in enumerate(membrane_potentials):
    if i==25:
        plt.plot(time_vector, membrane_potential)

plt.xlabel('Time (ms)')
plt.ylabel('Membrane Potential (mV)')
plt.title('Membrane Potentials of All Nodes Over Time')
plt.xlim(0, 5)
# plt.legend([f'Node {i}' for i in range(1, len(membrane_potentials) + 1)], loc='upper left')
plt.show()

# for current in transmembrane_current:
#     plt.plot(time_vector, current)

# plt.xlabel('Time (ms)')
# plt.ylabel('Transmembrane Current (nA)')
# plt.title('Transmembrane Current Over Time for Each Segment')
# plt.legend([f'Node {i}' for i in range(1, len(transmembrane_current) + 1)], bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.tight_layout()
# plt.show()

# for i, membrane_potential in enumerate(membrane_potentials):
#     if i % 2 == 0:  # Check if the index is a multiple of 10
#         plt.plot(time_vector, membrane_potential, label=f'Node {i+1}')

# plt.xlabel('Time (ms)')
# plt.ylabel('Membrane Potential (mV)')
# plt.title('Membrane Potentials of Every 10th Node Over Time')
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.tight_layout()
# plt.show()
# thresholds = []
# diams = np.linspace(2, 20, 10)

# for diam in diams:
#     thresh, _, _ = run(diam, 1*mm, pw=0.1)
#     thresholds.append(thresh)
    
# plt.plot(diams, thresholds); plt.xticks(diams)
# plt.ylabel('|threshold current| (mA)'); plt.xlabel('fiber diameter (μm)')
# plt.show() 