import argparse
import os
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


def plot(experiment_root_dir: str,
         smooth: int,
         start_step: int,
         end_step: int,
         width: int,
         height: int,
         color: str,
         save_as: str) -> None:

    # Create plot path if it does not exist
    plot_path: str = os.path.join('out', experiment_root_dir, 'plot')
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)

    # Load history csv
    data = pd.read_csv(os.path.join('out', experiment_root_dir, 'info', 'history.csv'))
    data = pd.DataFrame(data)

    # Change axis formatter
    ax = matplotlib.pyplot.gca()
    formatter = matplotlib.ticker.FuncFormatter(lambda x, pos: '%1.1fM' % (x * 1e-6) if x >= 1e6 else '%1dK' % (
            x * 1e-3) if x >= 1e3 else '%1d' % x)
    ax.xaxis.set_major_formatter(formatter)

    if end_step == 0:
        end_step = data["step"].values[-1]

    # Filter start and end step
    data = data[data["step"] >= start_step]
    data = data[data["step"] < end_step]

    # Set std limits
    data['std_bottom'] = data["mean"] - data["std"]
    data['std_top'] = data["mean"] + data["std"]

    if smooth > 0:
        # Smooth mean and std
        data['mean'] = data['mean'].rolling(window=smooth, win_type='triang',
                                            min_periods=1).mean()
        data['std_bottom'] = data['std_bottom'].rolling(window=smooth, win_type='triang',
                                                        min_periods=1).mean()
        data['std_top'] = data['std_top'].rolling(window=smooth, win_type='triang',
                                                  min_periods=1).mean()

    # Plot data
    plt.plot(data["step"], data["mean"], color=color, linewidth=1.5, alpha=1)
    plt.fill_between(data["step"], data['std_bottom'], data['std_top'], color=color, linewidth=1,
                     alpha=0.1)

    # Configure figure
    fig = plt.gcf()
    fig.set_size_inches(width, height)

    # Configure plot
    ax.grid(color='gray', linestyle='-', linewidth=1, alpha=0.2)
    ax.set_xlabel("Steps", fontsize=12)
    ax.set_ylabel("Episodic Return", fontsize=12)

    # Save plot
    plt.savefig(os.path.join(plot_path, f'episodic_return.{save_as}'), bbox_inches='tight')
    print(f"Plot saved to: {os.path.join(plot_path, f'episodic_return.{save_as}')}")
    plt.show()


def parse_arguments():
    arg = argparse.ArgumentParser()
    arg.add_argument("--experiment-root-dir", type=str, required=True, help="Experiment root directory.")
    arg.add_argument("--smooth", type=int, default=0, required=False, help="Smooth plot.")
    arg.add_argument("--start-step", type=int, default=0, required=False, help="Start plot step.")
    arg.add_argument("--end-step", type=int, default=0, required=False, help="End plot step.")
    arg.add_argument("--width", type=int, default=10, required=False, help="Plot width.")
    arg.add_argument("--height", type=int, default=6, required=False, help="Plot height.")
    arg.add_argument("--color", type=str, default='red', required=False, help="Plot color (red, blue, green).")
    arg.add_argument("--save-as", type=str, default='pdf', required=False, help="Save plot format (pdf, png, jpg).")

    return vars(arg.parse_args())


if __name__ == '__main__':
    args = parse_arguments()

    plot(**args)
