from neuron import h, gui
from neuron.units import ms, mV
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import pandas as pd
# Reciprocity, make the recording electrode the stimulating current and calculate
# Make it so that you know the axon node locations in the x-y-z plane and then just apply the voltages through interpolation at those locations
# put a point source at the recording electrode, calculate the voltage at each node, and sum and multiple by current to get the recording electrode voltage value
h.load_file('stdrun.hoc')

h.tstop = 10 # [ms]: simulation
h.dt = 0.01 # [ms]: timestep

np.random.seed(0)

potential = []
potential_list = []

def generate_axon_diameters(num_axons, mean_diameter, std_diameter):
    diameters = np.random.normal(loc=mean_diameter, scale=std_diameter, size=num_axons)
    return diameters

# Generate axon diameters
n_axons = 10
mean_diameter_small = 13  # µm
mean_diameter_large = 17  # µm
std_dev = 2
axon_diameters_small = generate_axon_diameters(n_axons, mean_diameter_small, std_dev)
axon_diameters_small = np.round(axon_diameters_small, 2)
axon_diameters_large = generate_axon_diameters(n_axons, mean_diameter_large, std_dev)
axon_diameters_large = np.round(axon_diameters_large, 2)

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

potential_tot = []
for D in np.concatenate([axon_diameters_small, axon_diameters_large]):

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
    amp = 2 # [mA]
    e2f = 1 # [mm]
    dur = 0.1 # [ms]
    
    nodes = [h.Section(name=f'node[{i}]') for i in range(n_nodes)]

    for node_ind, node in enumerate(nodes):
        node.nseg = nseg
        node.diam = 0.6*D
        node.L = L
        node.Ra = rhoa*((L+inl)/L) # left this in here since it is a fn(*other params)
        node.cm = cm
        # node.insert('sweeney')
        node.insert('extracellular')
        for seg in node:
            seg.extracellular.e = 0
        if node_ind>0:
            node.connect(nodes[node_ind-1](1))

    # INSTRUMENTATION - STIMULATION/RECORDING
    dummy = h.Section(name='dummy')
    # dummy.insert('extracellular')
    e_stim_obj = h.IClamp(nodes[0](0.5)) #put IClamp at end of axon?
    # e_stim_obj = h.IClamp(dummy(0.5)) # puts the stim halfway along the length
    e_stim_obj.delay = delay
    e_stim_obj.amp = amp 
    e_stim_obj.dur = dur

    vol_mem = [h.Vector().record(sec(0.5)._ref_v) for sec in nodes]
    tvec = h.Vector().record(h._ref_t)
    im_rec = [h.Vector().record(sec(0.5)._ref_i_membrane) for sec in nodes]

    # Determine the location of each node for interpolation
    node_positions = np.linspace(-n_nodes//2, n_nodes//2, n_nodes) * (1e-5*(L + inl))
    # print(node_positions)
    # print("node positions = {node_positions}".format(node_positions = node_positions))
    # Interpolate electric field data to find potential at each node
    points = electric_field_data[:, :2]  # x, y positions
    values = electric_field_data[:, 2]   # Electric potential
    print(values)
    node_potentials = griddata(points, values, (node_positions, np.zeros_like(node_positions)), method='linear')

    V1_sum = []
    V2_sum = []
    V3_sum = []


    def update_field():
        V1_rec = []
        V2_rec = []
        V3_rec = []
        for node_ind, node in enumerate(nodes):
            x_loc_1 = 1e-3*(inl*node_ind - 40000) #60mm from end of axon
            x_loc_2 = 1e-3*(inl*node_ind - 36000) #4mm from center electrode
            x_loc_3 = 1e-3*(inl*node_ind - 44000) #4mm from center electrode
            # Problem 2.3: point current source location
            x_loc_4 = 1e-3*(inl*node_ind - 30000) #halfway between axon end and electrode
            
            r_1 = np.sqrt(x_loc_1**2 + e2f**2) # [mm]
            r_2 = np.sqrt(x_loc_2**2 + e2f**2) # [mm]
            r_3 = np.sqrt(x_loc_3**2 + e2f**2) # [mm]
            r_4 = np.sqrt(x_loc_4**2 + e2f**2) # [mm]
            
            # node(0.5).e_extracellular = e_stim_obj.i/(4*sigma_e*np.pi*r_4)
            node(0.5).e_extracellular = node_potentials[node_ind]

            V1 = (node(0.5).i_membrane*1e-8*(0.6*np.pi*D*L))/(4*sigma_e*np.pi*r_1)
            V1_rec.append(V1)
            V2 = (node(0.5).i_membrane*1e-8*(0.6*np.pi*D*L))/(4*sigma_e*np.pi*r_2)
            V2_rec.append(V2)
            V3 = (node(0.5).i_membrane*1e-8*(0.6*np.pi*D*L))/(4*sigma_e*np.pi*r_3)
            V3_rec.append(V3)
        
        V1_sum.append(np.sum(V1_rec))
        V2_sum.append(np.sum(V2_rec)) 
        V3_sum.append(np.sum(V3_rec))     
    
    t_array = []

    # time integrate with constant time step
    def my_advance():
        update_field()
        global voltage
        voltage = np.array(V1_sum) - ((np.array(V2_sum) + np.array(V3_sum))/2)
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
    potential.append(voltage)

    

small_total_voltage = []
large_total_voltage = []

small_total_voltage = np.sum(potential[:45], axis=0)
large_total_voltage = np.sum(potential[45:], axis=0)
total_voltage = np.sum(potential, axis = 0)
# potential_list.append(total_voltage)


plt.plot(t_array, total_voltage, label = "total voltage")
# plt.plot(t_array, small_total_voltage, label = "small diameter fibers")
# plt.plot(t_array, potential_list, label = "large diameter fibers")
plt.xlabel("Time (ms)")
plt.ylabel("Transmembrane Potential (mV)")
plt.title("Compound Nerve Action Potential (CNAP)")
# plt.legend(["total_voltage", "small diameter fibers", "large diameter fibers"], loc="lower right")
plt.show()

