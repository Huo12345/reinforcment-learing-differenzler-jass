import os.path

import argparse
import pandas as pd


def plot_charts(args: argparse.Namespace) -> None:
    """
    Plots the training charts

    :param args: Command line arguments
    :return:
    """
    # Renames variables
    data_folder = args.data
    results_folder = args.result
    title = args.title
    show_results = args.show

    # Loads data
    win_rate = pd.read_csv(os.path.join(data_folder, 'win.csv'))
    points = pd.read_csv(os.path.join(data_folder, 'pts.csv'))

    # Shifts axis to be more readable
    win_rate['reward'] *= 100
    points['reward'] *= -157

    # Creates the winrate plot
    win_plt = win_rate.plot(x="episode", y="reward")
    win_plt.set_title(title)
    win_plt.set_xlabel('Episode')
    win_plt.set_ylabel('Win rate in %')
    win_plt.grid()
    if show_results:
        win_plt.figure.show()
    if results_folder is not None:
        win_plt.figure.savefig(os.path.join(results_folder, 'win.png'))

    # Creates the average points off plot
    pts_plt = points.plot(x="episode", y="reward")
    pts_plt.set_title(title)
    pts_plt.set_xlabel('Episode')
    pts_plt.set_ylabel('Points off on avg')
    pts_plt.grid()
    if show_results:
        pts_plt.figure.show()
    if results_folder is not None:
        pts_plt.figure.savefig(os.path.join(results_folder, 'pts.png'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Plot creator Differaenzler")
    parser.add_argument(
        '--data',
        type=str,
        required=True,
    )

    parser.add_argument(
        '--title',
        type=str,
        required=True,
    )

    parser.add_argument(
        '--result',
        type=str,
        default=None,
    )
    parser.add_argument(
        '--show',
        type=bool,
        default=False,
    )
    cli_args = parser.parse_args()
    plot_charts(cli_args)
