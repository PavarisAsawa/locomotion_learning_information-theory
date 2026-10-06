import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import animation
from pathlib import Path

def plot_pca_trajectory_3d(weight_pcs, pca, figsize=(8, 6), save_path=None):

    n_steps = weight_pcs.shape[0]
    time_steps = np.arange(n_steps)

    time_cmap = LinearSegmentedColormap.from_list("early_late", ["#2166AC", "#8C6BB1", "#B2182B"])

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(projection="3d")


    ax.computed_zorder = False
    # -------------------------------------------------------- Data ---
    # Underlying trajectory line
    ax.plot(weight_pcs[:, 0], weight_pcs[:, 1], weight_pcs[:, 2],color="gray", linewidth=1.2, alpha=0.4, zorder=0)

    # Time-colored scatter points
    sc = ax.scatter(
        weight_pcs[:, 0], weight_pcs[:, 1], weight_pcs[:, 2],
        c=time_steps, cmap=time_cmap, s=14, marker="o",
        linewidths=0, alpha=0.9, zorder=1,
    )

    # ---------------------------------------- First / Last markers ---
    first, last = weight_pcs[0], weight_pcs[-1]
    
    ax.scatter(first[0], first[1], first[2],c=[time_cmap(0.0)], s=100, marker="s",edgecolors="white", linewidths=1.5, depthshade=False,label="start (t=0)", zorder=10)
    
    ax.scatter(last[0], last[1], last[2],c=[time_cmap(1.0)], s=200, marker="*",edgecolors="white", linewidths=1.5, depthshade=False,label=f"end (t={n_steps - 1})", zorder=11)

    # ------------------------------------------------------ Labels ---
    evr = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({evr[0]:.1%})")
    ax.set_ylabel(f"PC2 ({evr[1]:.1%})")
    ax.set_zlabel(f"PC3 ({evr[2]:.1%})")

    ax.legend(loc="upper left", frameon=False)

    # ---------------------------------------------------- Colorbar ---
    cbar = fig.colorbar(sc, ax=ax, pad=0.1, shrink=0.7)
    cbar.set_label("timestep")
    cbar.outline.set_visible(False)

    fig.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        
    plt.show()
    
    return fig, ax

# =====================================================================
# Usage Example:
# =====================================================================
# fig, ax = plot_pca_trajectory_3d(weight_pcs, pca, save_path="pca_traj.png")



def animate_pca_trajectory_3d(
    weight_pcs, 
    pca, 
    stride=5, 
    trail=80, 
    fps=20, 
    rotate=40, 
    out_dir=None, 
    filename="pca_trajectory"
):
    """
    Animates a 3D PCA trajectory and saves it as a GIF (and MP4 if FFmpeg is available).

    Args:
        weight_pcs (np.ndarray): The PCA-transformed data of shape (n_samples, 3).
        pca (sklearn.decomposition.PCA): The fitted PCA model for variance labels.
        stride (int): Keep every Nth sample (fewer = faster rendering/playback).
        trail (int): How many past samples stay visible behind the head.
        fps (int): Frames per second for the output file.
        rotate (float): Total azimuth sweep in degrees (0 disables rotation).
        out_dir (str or Path, optional): Directory to save outputs. Defaults to current dir.
        filename (str): Base name for the saved files (without extension).
        
    Returns:
        dict: A dictionary containing the paths to the saved animation files.
    """
    if out_dir is None:
        out_dir = Path.cwd()
    else:
        out_dir = Path(out_dir)
        
    out_dir.mkdir(parents=True, exist_ok=True)
    
    n = weight_pcs.shape[0]
    frames = np.arange(0, n, stride)

    # Custom Colormap: Blue (early) -> Purple -> Red (late)
    time_cmap = LinearSegmentedColormap.from_list(
        "early_late", ["#2166AC", "#8C6BB1", "#B2182B"]
    )

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(projection="3d")
    ax.computed_zorder = False

    # Faint full cloud so the shape of the space stays visible the whole time.
    ax.scatter(
        weight_pcs[:, 0], weight_pcs[:, 1], weight_pcs[:, 2],
        c="0.78", s=7, linewidths=0, depthshade=False, zorder=0
    )

    # Empty scatter objects to be updated in the animation loop
    trail_scatter = ax.scatter([], [], [], s=30, linewidths=0, depthshade=False, zorder=5)
    head_scatter = ax.scatter([], [], [], s=340, marker="o", edgecolors="white",
                              linewidths=1.8, depthshade=False, zorder=20)

    # Start and End markers
    first, last = weight_pcs[0], weight_pcs[-1]
    
    ax.scatter(
        first[0], first[1], first[2], 
        c=[time_cmap(0.0)], s=200, marker="s",
        edgecolors="white", linewidths=1.5, depthshade=False,
        label="start (t=0)", zorder=10
    )
    
    ax.scatter(
        last[0], last[1], last[2], 
        c=[time_cmap(1.0)], s=420, marker="*",
        edgecolors="white", linewidths=1.5, depthshade=False,
        label=f"end (t={n - 1})", zorder=11
    )

    # Labels and Limits
    evr = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({evr[0]:.1%})")
    ax.set_ylabel(f"PC2 ({evr[1]:.1%})")
    ax.set_zlabel(f"PC3 ({evr[2]:.1%})")
    
    # Locking limits prevents the camera from zooming/jumping during playback
    ax.set_xlim(weight_pcs[:, 0].min(), weight_pcs[:, 0].max())
    ax.set_ylim(weight_pcs[:, 1].min(), weight_pcs[:, 1].max())
    ax.set_zlim(weight_pcs[:, 2].min(), weight_pcs[:, 2].max())
    ax.legend(loc="upper left", frameon=False)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=time_cmap, norm=plt.Normalize(0, n - 1))
    cbar = fig.colorbar(sm, ax=ax, pad=0.1, shrink=0.7)
    cbar.set_label("timestep")
    cbar.outline.set_visible(False)

    # On-screen clock
    clock = ax.text2D(0.98, 0.02, "", transform=ax.transAxes,
                      ha="right", va="bottom", fontsize=11, color="0.35")

    azim0 = ax.azim

    # The update function called per frame
    def update(i):
        lo = max(0, i - trail)
        seg = weight_pcs[lo:i + 1]
        
        # Update trail
        trail_scatter._offsets3d = (seg[:, 0], seg[:, 1], seg[:, 2])
        trail_scatter.set_color(time_cmap(np.linspace(lo, i, len(seg)) / max(n - 1, 1)))
        trail_scatter.set_alpha(0.85)

        # Update head
        pt = weight_pcs[i]
        head_scatter._offsets3d = ([pt[0]], [pt[1]], [pt[2]])
        head_scatter.set_color([time_cmap(i / max(n - 1, 1))])

        # Rotate camera
        if rotate:
            ax.view_init(elev=ax.elev, azim=azim0 + rotate * i / max(n - 1, 1))

        clock.set_text(f"t = {i}")
        return trail_scatter, head_scatter, clock

    # Generate animation
    anim = animation.FuncAnimation(fig, update, frames=frames,
                                   interval=1000 / fps, blit=False)

    saved_files = {}

    # Save GIF
    gif_path = out_dir / f"{filename}.gif"
    anim.save(gif_path, writer=animation.PillowWriter(fps=fps), dpi=90)
    print(f"Wrote {gif_path}")
    saved_files['gif'] = gif_path

    # Save MP4 (MP4 needs ffmpeg on PATH; skipped cleanly when it is not installed)
    if animation.FFMpegWriter.isAvailable():
        mp4_path = out_dir / f"{filename}.mp4"
        anim.save(mp4_path, writer=animation.FFMpegWriter(fps=fps, bitrate=2400), dpi=140)
        print(f"Wrote {mp4_path}")
        saved_files['mp4'] = mp4_path
    else:
        print("FFMpeg not found - GIF only. `conda install ffmpeg` or add it to PATH for MP4.")

    plt.close(fig)
    return saved_files

# =====================================================================
# Usage Example:
# =====================================================================
# result = animate_pca_trajectory_3d(weight_pcs, pca, stride=5, trail=100, rotate=60)
# print("Animation saved to:", result['gif'])