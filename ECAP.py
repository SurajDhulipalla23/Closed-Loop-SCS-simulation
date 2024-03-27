from neuron import h, gui
import matplotlib.pyplot as plt

# Initialize the model
h.load_file('stdrun.hoc')

# Create a soma section
soma = h.Section(name='soma')
soma.L = 20  # Length of the soma in micrometers
soma.diam = 20  # Diameter of the soma in micrometers
soma.Ra = 100  # Axial resistance in Ohm*cm
soma.insert('hh')  # Insert Hodgkin-Huxley ion channels

# Create a stimulator
stim = h.IClamp(soma(0.5))  # Place it in the middle of the soma
stim.delay = 100  # Start of stimulation in ms
stim.dur = 100  # Duration of stimulation in ms
stim.amp = 0.1  # Stimulation amplitude in nA

# Record time, voltage, and stimulation current
t = h.Vector().record(h._ref_t)  # Time vector
v = h.Vector().record(soma(0.5)._ref_v)  # Membrane potential vector
i_stim = h.Vector().record(stim._ref_i)  # Stimulation current

# Simulation parameters
h.tstop = 300  # ms

# Run the simulation
h.run()



# Plotting
plt.figure(figsize=(10, 8))
plt.subplot(2, 1, 1)
plt.plot(t, v, label='Membrane Potential (mV)')
plt.ylabel('Voltage (mV)')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(t, i_stim, label='Stimulation Current (nA)', color='red')
plt.xlabel('Time (ms)')
plt.ylabel('Current (nA)')
plt.legend()

plt.show()
