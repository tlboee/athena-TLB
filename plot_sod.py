import sys
import numpy as np
import matplotlib.pyplot as plt

# athena_read.py aus dem Athena++-Paket laden
sys.path.insert(0, 'vis/python')
import athena_read

# Zwei Zeitschritte laden: t=0 (Anfang) und t=Ende
data_init = athena_read.tab('Sod.block0.out1.00000.tab')
data_final = athena_read.tab('Sod.block0.out1.00020.tab')

x_init  = data_init['x1v']
x_final = data_final['x1v']

fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
fig.suptitle('Sod Schocktube: t=0 (blau) vs. t=Ende (orange)', fontsize=13)

axes[0].plot(x_init,  data_init['rho'],   label='t=0')
axes[0].plot(x_final, data_final['rho'],  label='t=Ende')
axes[0].set_ylabel('Dichte ρ')
axes[0].legend()

axes[1].plot(x_init,  data_init['press'])
axes[1].plot(x_final, data_final['press'])
axes[1].set_ylabel('Druck p')

axes[2].plot(x_init,  data_init['vel1'])
axes[2].plot(x_final, data_final['vel1'])
axes[2].set_ylabel('Geschwindigkeit v₁')
axes[2].set_xlabel('Position x')

plt.tight_layout()
plt.savefig('sod_plot.png', dpi=150)
plt.show()
print("Plot gespeichert als sod_plot.png")