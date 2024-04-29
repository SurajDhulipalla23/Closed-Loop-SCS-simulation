from neuron import h, gui
from neuron.units import ms, mV
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.interpolate import interp1d
import pandas as pd
from scipy.stats import norm

h.load_file('stdrun.hoc')

h.tstop = 25 # [ms]: simulation
h.dt = 0.01 # [ms]: timestep

np.random.seed(0)

potential = []
potential_list = []

def generate_axon_diameters(num_axons, mean_diameter, std_diameter):
    diameters = np.random.normal(loc=mean_diameter, scale=std_diameter, size=num_axons)
    return diameters

# Generate axon diameters
n_axons = 5
mean_diameter_small = 9  # µm
std_dev = 2
axon_diameters_a = generate_axon_diameters(n_axons, mean_diameter_small, std_dev)
axon_diameters_a = np.round(axon_diameters_a, 2)
axon_diameters_b = generate_axon_diameters(n_axons, mean_diameter_small, std_dev)
axon_diameters_b = np.round(axon_diameters_b, 2)
axon_diameters_a = generate_axon_diameters(n_axons, mean_diameter_small, std_dev)
axon_diameters_a = np.round(axon_diameters_a, 2)
axon_diameters_c = generate_axon_diameters(n_axons, mean_diameter_small, std_dev)
axon_diameters_c = np.round(axon_diameters_b, 2)

file_name = 'ElectricPotential3D_far.csv'
electric_field_data = np.genfromtxt(file_name, delimiter=',', skip_header=9)
df = pd.DataFrame(electric_field_data, columns=['x-value', 'y-value', 'z-value', 'E-field'])

file_name_a = 'FiberA_far.csv'
electric_field_data_a = np.genfromtxt(file_name_a, delimiter=',', skip_header=9)

file_name_b = 'FiberB_far.csv'
electric_field_data_b = np.genfromtxt(file_name_b, delimiter=',', skip_header=9)

file_name_c = 'FiberC_far.csv'
electric_field_data_c = np.genfromtxt(file_name_c, delimiter=',', skip_header=9)

# Convert x, y, z values from meters to millimeters by multiplying by 1000
df['x-value'] *= 1000
df['y-value'] *= 1000
df['z-value'] *= 1000

# Assuming E-field is directly proportional to the distance from the origin and its direction
# Calculate vector components for 3D
magnitude = np.sqrt(df['x-value']**2 + df['y-value']**2 + df['z-value']**2)
df['Ex'] = df['E-field'] * df['x-value'] / magnitude
df['Ey'] = df['E-field'] * df['y-value'] / magnitude
df['Ez'] = df['E-field'] * df['z-value'] / magnitude

# Handle cases where x, y, and z are 0 to avoid division by zero
df.fillna(0, inplace=True)

# Plotting in 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
q = ax.quiver(df['x-value'], df['y-value'], df['z-value'], df['Ex'], df['Ey'], df['Ez'], length=5, normalize=True)
ax.set_xlabel('X position (mm)')
ax.set_ylabel('Y position (mm)')
ax.set_zlabel('Z position (mm)')
plt.title('3D Electric Field Visualization')

plt.show()

# Set Gaussian distribution parameters
x_mean = 0.0015  # meters (converted to mm when used)
y_mean = 0.00525    # meters (converted to mm when used)
std_dev = 0.001  # meters (1000 mm to spread around mean)

# Generate x, y positions for each axon according to a Gaussian distribution
num_axons = len(np.concatenate([axon_diameters_a, axon_diameters_b, axon_diameters_c]))
x_positions = norm.rvs(loc=x_mean, scale=std_dev, size=num_axons)
y_positions = norm.rvs(loc=y_mean, scale=std_dev, size=num_axons)

# Define axons and set positions
axon_list = []
for axon_index, D in enumerate(np.concatenate([axon_diameters_a, axon_diameters_b, axon_diameters_c])):

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
    e2f = 5 # [mm]
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
        
        # Assign 3D position to each node: axons are vertically aligned in z-direction
        node_x.append(x_positions[axon_index])
        node_y.append(y_positions[axon_index])
        node_z.append(node_ind * inl * (1e-6))  # internodal length converted to mm

    # print("X positions: {}".format(node_x)) # X position not working in gaussian
    # print("Y positions: {}".format(node_y)) # Y position not working in gaussian
    # print("Z positions: {}".format(node_z))
    

    axon_list.append(nodes)


    vol_mem = [h.Vector().record(sec(0.5)._ref_v) for sec in nodes]
    tvec = h.Vector().record(h._ref_t)
    im_rec = [h.Vector().record(sec(0.5)._ref_i_membrane) for sec in nodes]

    # Determine the location of each node for interpolation
    # print("node positions = {node_positions}".format(node_positions = node_positions))
    # Interpolate electric field data to find potential at each node
    node_positions = np.column_stack((node_x,node_y, node_z))
    # points = electric_field_data[:, :3]  # x, y, z positions
    # values = electric_field_data[:, 3]   # Electric potential
    # node_potentials = griddata(points, values, node_positions, method='linear')
    if axon_index < 10:
        points_a = electric_field_data_a[:, 0]  # x positions
        values_a = electric_field_data_a[:, 1]   # Electric potential
        interp_a = interp1d(points_a, values_a, kind='linear', bounds_error=False, fill_value='extrapolate')
        node_potentials = interp_a(node_z)
        node_potentials = np.nan_to_num(node_potentials, nan=0.0)
    if axon_index < 20:
        points_b = electric_field_data_b[:, 0]  # x positions
        values_b = electric_field_data_b[:, 1]   # Electric potential
        interp_b = interp1d(points_b, values_b, kind='linear', bounds_error=False, fill_value='extrapolate')
        node_potentials = interp_b(node_z)
        node_potentials = np.nan_to_num(node_potentials, nan=0.0)
    else:
        points_c = electric_field_data_c[:, 0]  # x positions
        values_c = electric_field_data_c[:, 1]   # Electric potential
        interp_c = interp1d(points_c, values_c, kind='linear', bounds_error=False, fill_value='extrapolate')
        node_potentials = interp_a(node_z)
        node_potentials = np.nan_to_num(node_potentials, nan=0.0)

    print(node_potentials)
    V1_sum= []
    V2_sum = []
    V3_sum = []

    def update_field():
        V1_rec = []
        V2_rec = []
        V3_rec = []
        for node_ind, node in enumerate(nodes):
            x_loc_1 = 1e-3*(inl*node_ind - 40000) #35mm from end of axon
            x_loc_2 = 1e-3*(inl*node_ind - 36000) #4mm from center electrode
            x_loc_3 = 1e-3*(inl*node_ind - 44000) #4mm from center electrode
            # x_loc_1 = 1e-3*(inl*node_ind - 4000) #35mm from end of axon
            # x_loc_2 = 1e-3*(inl*node_ind - 3000) #4mm from center electrode
            # x_loc_3 = 1e-3*(inl*node_ind - 5000) #4mm from center electrode
            
            r_1 = np.sqrt(x_loc_1**2 + e2f**2) # [mm]
            r_2 = np.sqrt(x_loc_2**2 + e2f**2) # [mm]
            r_3 = np.sqrt(x_loc_3**2 + e2f**2) # [mm]
            
            # node(0.5).e_extracellular = e_stim_obj.i/(4*sigma_e*np.pi*r_4)
            node(0.5).e_extracellular = node_potentials[node_ind] * 6 #adjusting for the actual stimulation current
            # print("voltage: {}".format(node_potentials[node_ind]))
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
        # print("voltag:{}".format(voltage))
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


# print(potential)
small_total_voltage = []
large_total_voltage = []

small_total_voltage = np.sum(potential[:45], axis=0)
large_total_voltage = np.sum(potential[45:], axis=0)
total_voltage = np.sum(potential, axis = 0)
# potential_list.append(total_voltage)


plt.plot(t_array, total_voltage*(5e6), label = "total voltage")
# plt.plot(t_array, small_total_voltage, label = "small diameter fibers")
# plt.plot(t_array, potential_list, label = "large diameter fibers")
plt.xlabel("Time (ms)")
plt.ylabel("Transmembrane Potential (V)")
plt.title("Compound Nerve Action Potential (CNAP)")
# plt.legend(["total_voltage", "small diameter fibers", "large diameter fibers"], loc="lower right")
plt.show()

