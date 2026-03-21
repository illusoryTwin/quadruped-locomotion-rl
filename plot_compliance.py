import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("env0_compliance_log.csv")

fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
fig.suptitle("Go2 Compliant Stance – Forces & Deformations (Last Training Run: 2026-03-15)",
             fontsize=14, fontweight="bold")

# --- Applied Z-Force ---
ax = axes[0]
ax.plot(df["sim_time"], df["force_z"], linewidth=0.4, color="tab:red", alpha=0.8)
ax.set_ylabel("Applied Force Z  [N]", fontsize=11)
ax.set_title("External Sinusoidal Force (Z-axis)", fontsize=12)
ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
ax.grid(True, alpha=0.3)

# --- Deformation Z (dominant axis) ---
ax = axes[1]
ax.plot(df["sim_time"], df["x_def_z"], linewidth=0.4, color="tab:blue", alpha=0.8)
ax.set_ylabel("Deformation Z  [m]", fontsize=11)
ax.set_title("Cartesian Deformation – Z (Vertical)", fontsize=12)
ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
ax.grid(True, alpha=0.3)

# --- Deformation X & Y ---
ax = axes[2]
ax.plot(df["sim_time"], df["x_def_x"], linewidth=0.4, color="tab:green", alpha=0.8, label="x_def_x")
ax.plot(df["sim_time"], df["x_def_y"], linewidth=0.4, color="tab:orange", alpha=0.8, label="x_def_y")
ax.set_ylabel("Deformation  [m]", fontsize=11)
ax.set_xlabel("Simulation Time  [s]", fontsize=11)
ax.set_title("Cartesian Deformation – X & Y (Lateral)", fontsize=12)
ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
ax.legend(loc="upper left", fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("env0_compliance_plot_latest.png", dpi=150, bbox_inches="tight")
print("Saved: env0_compliance_plot_latest.png")

# --- Also create a zoomed view of the last 200s to see detail ---
fig2, axes2 = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
fig2.suptitle("Zoomed: Last 200s of Training – Force vs Deformation Response",
              fontsize=14, fontweight="bold")

mask = df["sim_time"] >= (df["sim_time"].max() - 200)
df_tail = df[mask]

ax = axes2[0]
ax.plot(df_tail["sim_time"], df_tail["force_z"], linewidth=0.6, color="tab:red")
ax.set_ylabel("Force Z  [N]", fontsize=11)
ax.set_title("Applied Force (Z)", fontsize=12)
ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
ax.grid(True, alpha=0.3)

ax = axes2[1]
ax.plot(df_tail["sim_time"], df_tail["x_def_z"], linewidth=0.6, color="tab:blue", label="x_def_z")
ax.plot(df_tail["sim_time"], df_tail["x_def_x"], linewidth=0.6, color="tab:green", label="x_def_x")
ax.plot(df_tail["sim_time"], df_tail["x_def_y"], linewidth=0.6, color="tab:orange", label="x_def_y")
ax.set_ylabel("Deformation  [m]", fontsize=11)
ax.set_xlabel("Simulation Time  [s]", fontsize=11)
ax.set_title("MSD Deformation Response", fontsize=12)
ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
ax.legend(loc="upper left", fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("env0_compliance_zoom_latest.png", dpi=150, bbox_inches="tight")
print("Saved: env0_compliance_zoom_latest.png")

# Print some stats
print(f"\n--- Data Summary ---")
print(f"Total steps: {len(df)}")
print(f"Sim time range: {df['sim_time'].min():.2f} – {df['sim_time'].max():.2f} s")
print(f"Force Z range: {df['force_z'].min():.2f} to {df['force_z'].max():.2f} N")
print(f"Deformation Z range: {df['x_def_z'].min():.4f} to {df['x_def_z'].max():.4f} m")
print(f"Deformation X range: {df['x_def_x'].min():.4f} to {df['x_def_x'].max():.4f} m")
print(f"Deformation Y range: {df['x_def_y'].min():.4f} to {df['x_def_y'].max():.4f} m")
