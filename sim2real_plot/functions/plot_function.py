import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from pathlib import Path
import matplotlib


def _ensure_ffmpeg():
    """
    Makes matplotlib's FFMpegWriter usable without a system-wide ffmpeg install.

    Checks PATH first; if that fails, falls back to the static binary shipped by
    the `imageio-ffmpeg` wheel and registers it in rcParams.

    Returns:
        bool: True if an MP4 writer is available.
    """
    if animation.FFMpegWriter.isAvailable():
        return True

    try:
        import imageio_ffmpeg
    except ImportError:
        return False

    matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
    return animation.FFMpegWriter.isAvailable()




import numpy as np
import matplotlib.pyplot as plt

def plot_joints_trajectory(trajectories, title_name, lim=[-1,1], labels=None):
    # Convert all inputs to numpy arrays
    trajs = [np.array(t) for t in trajectories]
    num_compares = len(trajs)
    
    if labels is None or len(labels) != num_compares:
        labels = [f"Trajectory {i}" for i in range(num_compares)]

    colors = ["#eb6834", "#1f77b4", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]
    linestyles = ["-", "--", "-.", ":", "-", "--", "-."]
    
    fig, axes = plt.subplots(nrows=5, ncols=4, figsize=(24, 18))
    fig.suptitle(title_name, fontsize=20, fontweight='bold', y=0.92)    
    axes = axes.flatten()
    plt.subplots_adjust(hspace=0.35)
    
    # Plot data for all 19 joints
    for i in range(19):
        # Shift the axis index forward by 1 if we reach the legend slot (index 3)
        ax_idx = i if i < 3 else i + 1
        ax = axes[ax_idx]
        
        # Loop through each trajectory passed into the function
        for j in range(num_compares):
            c = colors[j % len(colors)]
            ls = linestyles[j % len(linestyles)]
            
            ax.plot(trajs[j][:, i], color=c, linestyle=ls)
            
        ax.set_title(f"Joint {i}", fontsize=12, pad=1, alpha=0.7)
        ax.set_ylim(lim[0], lim[1])
        
    # Legend in the top-right slot (index 3)
    legend_ax = axes[3]                
    legend_ax.axis("off")
    
    # Dynamically generate legend handles for N items
    handles = [
        plt.Line2D([], [], 
                   color=colors[j % len(colors)], 
                   ls=linestyles[j % len(linestyles)], 
                   lw=2.0, 
                   label=labels[j]) 
        for j in range(num_compares)
    ]
    
    legend_ax.legend(handles=handles, loc="center", frameon=False, fontsize=12, handlelength=2.8, labelspacing=1.0)
    
    plt.show()
    return fig, axes

def plot_joint_trajectory(COMPARE0, title_name, label, lim=[-1,1]):
    COM0 = np.array(COMPARE0)
    T = np.arange(COM0.shape[0])

    COLOR = "#eb6834"
    fig, axes = plt.subplots(nrows=5, ncols=4, figsize=(24, 18))
    fig.suptitle(title_name, fontsize=20, fontweight='bold', y=0.92)    
    axes = axes.flatten()
    plt.subplots_adjust(hspace=0.35)

    for i in range(19):
        ax = axes[i]

        ax.plot(COM0[:,i], label=f'joint {i}', color=COLOR, )
        ax.set_title(f"Joint {i}", fontsize=12,  pad=1, alpha=0.7)
        ax.set_ylim(lim[0], lim[1])
        
    legend_ax = axes[-1]                 # legend in the spare slot
    legend_ax.axis("off")
    series = [(None, "-",  1.6, label)]
    handles = [plt.Line2D([], [], color=COLOR, ls=ls, lw=lw + 0.4, label=lab) for _, ls, lw, lab in series]
    legend_ax.legend(handles=handles, loc="center", frameon=False, fontsize=12,
                    handlelength=2.8, labelspacing=1.0)

    plt.show()
    return fig, axes

def plot_joint_index(COMPARE0, COMPARE1, joint_idx, title_name="", hz=50,
                     figsize=None, save_path=None,lim=[-1,1]):
    """
    Plots the full trajectory of one or more selected joints (static figure).

    Args:
        COMPARE0 (array): Position feedback, shape (T, n_joints).
        COMPARE1 (array): Motor command, shape (T, n_joints).
        joint_idx (int or list[int]): Which joint column(s) to plot.
        title_name (str): Figure title.
        hz (float): Control rate, used to label the x axis in seconds.
        figsize (tuple, optional): Defaults to a height that scales with joint count.
        save_path (str, optional): If given, the figure is written here.

    Returns:
        (fig, axes)
    """
    COM0 = np.array(COMPARE0)
    COM1 = np.array(COMPARE1)
    idx = [joint_idx] if np.isscalar(joint_idx) else list(joint_idx)

    t = np.arange(COM0.shape[0]) / hz
    COLOR = "#eb6834"

    if figsize is None:
        figsize = (12, 2.6 * len(idx) + 0.8)

    fig, axes = plt.subplots(nrows=len(idx), ncols=1, figsize=figsize, sharex=True)
    axes = np.atleast_1d(axes)
    if title_name:
        fig.suptitle(title_name, fontsize=15, fontweight="bold")

    for ax, j in zip(axes, idx):
        ax.plot(t, COM0[:, j], color=COLOR, lw=1.6, label="Position (feedback)")
        ax.plot(t, COM1[:, j], color=COLOR, lw=2.0, ls="dashed", label="Motor Command")
        ax.set_title(f"Joint {j}", fontsize=12, pad=2, alpha=0.7)
        ax.set_ylabel("value")
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylim(lim[0], lim[1])

    axes[-1].set_xlabel("time (s)")
    axes[0].legend(loc="upper right", frameon=False, fontsize=10)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")

    plt.show()
    return fig, axes


def animate_joint_index(
    COMPARE0,
    COMPARE1,
    joint_idx,
    title_name="",
    hz=50,
    window=100,
    stride=1,
    out_dir=None,
    filename="joint_signal",
    save_gif=False,
    fixed_ylim=True,
    dpi=140,
    bitrate=2400,
):
    """
    Animates selected joints as a scrolling real-time trace and saves it to video.

    Playback is real time: one timestep is 1/hz seconds, so the writer runs at
    `hz / stride` fps. The x axis holds a fixed `window` timesteps
    (window / hz seconds) and scrolls once the trace reaches the right edge.

    Args:
        COMPARE0 (array): Position feedback, shape (T, n_joints).
        COMPARE1 (array): Motor command, shape (T, n_joints).
        joint_idx (int or list[int]): Which joint column(s) to animate.
        title_name (str): Figure title.
        hz (float): Control rate in Hz (50 Hz -> 0.02 s per timestep).
        window (int): Number of timesteps visible at once (100 -> 2 s at 50 Hz).
        stride (int): Render every Nth timestep. >1 drops frames but keeps
            playback real time by lowering fps to match.
        out_dir (str or Path, optional): Output directory. Defaults to cwd.
        filename (str): Base name for the saved files (no extension).
        save_gif (bool): Also write a GIF. Off by default because a 50 fps GIF
            is very large; the MP4 is the real-time deliverable.
        fixed_ylim (bool): Lock y limits to the full-run range (True) or rescale
            to the visible window each frame (False).
        dpi (int): Output resolution of the MP4.
        bitrate (int): MP4 bitrate in kbit/s.

    Returns:
        dict: Paths of the files that were written, keyed 'mp4' / 'gif'.
    """
    COM0 = np.array(COMPARE0)
    COM1 = np.array(COMPARE1)
    idx = [joint_idx] if np.isscalar(joint_idx) else list(joint_idx)

    out_dir = Path.cwd() if out_dir is None else Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n = COM0.shape[0]
    frames = np.arange(0, n, stride)
    fps = hz / stride                      # keeps playback at wall-clock speed
    win_sec = window / hz
    t = np.arange(n) / hz

    COLOR = "#eb6834"
    fig, axes = plt.subplots(nrows=len(idx), ncols=1,
                             figsize=(12, 2.6 * len(idx) + 1.0), sharex=True)
    axes = np.atleast_1d(axes)
    if title_name:
        fig.suptitle(title_name, fontsize=15, fontweight="bold")

    lines0, lines1, heads0, heads1, readouts = [], [], [], [], []

    for ax, j in zip(axes, idx):
        (l0,) = ax.plot([], [], color=COLOR, lw=1.6, label="Position (feedback)")
        (l1,) = ax.plot([], [], color=COLOR, lw=2.0, ls="dashed", label="Motor Command")
        (h0,) = ax.plot([], [], "o", color=COLOR, ms=7, mec="white", mew=1.2)
        (h1,) = ax.plot([], [], "o", color=COLOR, ms=7, mfc="white", mew=1.6)

        ax.set_title(f"Joint {j}", fontsize=12, pad=2, alpha=0.7)
        ax.set_ylabel("value")
        ax.spines[["top", "right"]].set_visible(False)

        if fixed_ylim:
            lo = min(COM0[:, j].min(), COM1[:, j].min())
            hi = max(COM0[:, j].max(), COM1[:, j].max())
            pad = 0.08 * (hi - lo) if hi > lo else 0.1
            ax.set_ylim(lo - pad, hi + pad)

        txt = ax.text(0.995, 0.94, "", transform=ax.transAxes, ha="right", va="top",
                      fontsize=10, color="0.30", family="monospace")

        lines0.append(l0); lines1.append(l1)
        heads0.append(h0); heads1.append(h1)
        readouts.append(txt)

    axes[-1].set_xlabel("time (s)")
    axes[0].legend(loc="upper left", frameon=False, fontsize=10)

    clock = axes[0].text(0.005, 0.94, "", transform=axes[0].transAxes,
                         ha="left", va="bottom", fontsize=11, color="0.35")
    fig.tight_layout()

    def update(i):
        lo = max(0, i - window + 1)
        sl = slice(lo, i + 1)
        left = t[lo]

        for k, j in enumerate(idx):
            seg0, seg1 = COM0[sl, j], COM1[sl, j]
            lines0[k].set_data(t[sl], seg0)
            lines1[k].set_data(t[sl], seg1)
            heads0[k].set_data([t[i]], [COM0[i, j]])
            heads1[k].set_data([t[i]], [COM1[i, j]])
            readouts[k].set_text(f"pos {COM0[i, j]:+.3f}   cmd {COM1[i, j]:+.3f}")

            axes[k].set_xlim(left, left + win_sec)
            if not fixed_ylim:
                y_lo = min(seg0.min(), seg1.min())
                y_hi = max(seg0.max(), seg1.max())
                pad = 0.08 * (y_hi - y_lo) if y_hi > y_lo else 0.1
                axes[k].set_ylim(y_lo - pad, y_hi + pad)

        clock.set_text(f"t = {t[i]:6.2f} s   (step {i}/{n - 1})")
        return (*lines0, *lines1, *heads0, *heads1, *readouts, clock)

    anim = animation.FuncAnimation(fig, update, frames=frames,
                                   interval=1000 / fps, blit=False)

    saved_files = {}

    if _ensure_ffmpeg():
        mp4_path = out_dir / f"{filename}.mp4"
        anim.save(
            mp4_path,
            writer=animation.FFMpegWriter(
                fps=fps,
                bitrate=bitrate,
                # yuv420p + even dimensions keep the file playable in browsers,
                # PowerPoint and QuickTime, not just VLC.
                extra_args=["-pix_fmt", "yuv420p", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2"],
            ),
            dpi=dpi,
        )
        print(f"Wrote {mp4_path}")
        saved_files["mp4"] = mp4_path
    else:
        print("FFMpeg not found - falling back to GIF. Run "
              "`pip install imageio-ffmpeg` for a real-time MP4.")
        save_gif = True

    if save_gif:
        gif_path = out_dir / f"{filename}.gif"
        anim.save(gif_path, writer=animation.PillowWriter(fps=fps), dpi=90)
        print(f"Wrote {gif_path}")
        saved_files["gif"] = gif_path

    plt.close(fig)
    return saved_files


# =====================================================================
# Usage Example:
# =====================================================================
# plot_joint_index(COMPARE0, COMPARE1, [2, 5], title_name="ff-slopeup-grass-0")
# result = animate_joint_index(COMPARE0, COMPARE1, 2, hz=50, window=100,
#                              out_dir=SAVE_PATH, filename="joint2")
