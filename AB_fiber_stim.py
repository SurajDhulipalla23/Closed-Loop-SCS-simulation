from neuron import h, gui
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata



# Load NEURON extracellular stimulation mechanisms
h.load_file('stdrun.hoc')

# Define the myelinated fiber model class
class MyelinatedFiber:
    def __init__(self, fiber_length=1000, num_segments=10):
        self.segments = []
        self.build_fiber(fiber_length, num_segments)
        
    def build_fiber(self, fiber_length, num_segments):
        for _ in range(num_segments):
            seg = h.Section(name='seg')
            seg.L = fiber_length / num_segments  # Length of each segment
            seg.diam = 10  # Diameter in micrometers
            seg.nseg = 1  # Number of segments in this section
            seg.insert('extracellular')  # For extracellular stimulation
            seg.insert('hh')
            self.segments.append(seg)
            
    def apply_electric_field(self, field_data):
        """Apply extracellular electric field data to the fiber."""
        for seg, e_field in zip(self.segments, field_data):
            for seg_part in seg:
                seg_part.e_extracellular = e_field

# Initialize the myelinated fiber model
fiber = MyelinatedFiber(fiber_length=1000, num_segments=100)

# Example electric field data from COMSOL (in mV/mm), should be replaced with actual data
# This is a placeholder for time-varying electric field data per segment
electric_field_data = np.random.rand(100, 100)  # 100 time steps, 100 segments

# Simulation parameters
tstop = 100  # Simulation end time in ms
dt = 1  # Time step in ms

# Set up recording vectors
time = h.Vector().record(h._ref_t)
v_m = [h.Vector().record(seg(0.5)._ref_v) for seg in fiber.segments]
i_na = [h.Vector().record(seg(0.5)._ref_ina) for seg in fiber.segments]

# Run the simulation across time steps
for t_idx in range(int(tstop/dt)):
    # Update electric field for this time step across all segments
    fiber.apply_electric_field(electric_field_data[:, t_idx])
    
    # Advance the simulation
    h.fadvance()

# Convert time vector to a numpy array for plotting
# time_np = np.array(time)
plt.plot(time, np.array(v_m[25]), label=f'Segment {i+1}')

# # Plotting membrane potentials
# plt.figure(figsize=(10, 8))
# for i, v in enumerate(v_m):
#     plt.plot(time_np, np.array(v[25]), label=f'Segment {i+1}')
# plt.xlabel('Time (ms)')
# plt.ylabel('Membrane Potential (mV)')
# plt.title('Membrane Potential Over Time for Each Segment')
# plt.legend()
# plt.show()

# # Plotting transmembrane currents - replace i_na with your actual current vectors
# plt.figure(figsize=(10, 8))
# for i, current in enumerate(i_na):  # Replace i_na accordingly
#     plt.plot(time_np, np.array(current), label=f'Segment {i+1}')
# plt.xlabel('Time (ms)')
# plt.ylabel('Transmembrane Current (nA)')
# plt.title('Transmembrane Current Over Time for Each Segment')
# plt.legend()
# plt.show()
