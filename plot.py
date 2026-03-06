import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('statboticsData.csv')
winrate = df['winrate']
autoRP = df['rp_1_epa']
coralRP = df['rp_2_epa']
bargeRP = df['rp_3_epa']

X = df[["rp_1_epa", "rp_2_epa", "rp_3_epa"]]
y = df["winrate"]

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Create the 3D scatter plot
# c=winrate applies the winrate values to the color of the dots
# cmap='viridis' is a color map that transitions from dark purple (low) to bright yellow (high)
scatter = ax.scatter(autoRP, coralRP, bargeRP, c=winrate, cmap='viridis', s=50, alpha=0.8)

# Set the title and axis labels to match your photo
ax.set_title('Expected Winrate dependent on EPA')
ax.set_xlabel('autoRP')
ax.set_ylabel('coralRP')
ax.set_zlabel('bargeRP')

# Add a color bar legend to the side so you know what winrate the colors represent
cbar = fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=10, pad=0.1)
cbar.set_label('Winrate')

# Display the plot
plt.show()