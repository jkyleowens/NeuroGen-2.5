import argparse

from .simulation import add_simulation_parser, run_simulation
from .language.cli import add_language_parser, run_language


def main():
    parser = argparse.ArgumentParser(description="Continuous laminar spiking network simulation")
    commands = parser.add_subparsers(dest="command", required=True)
    add_simulation_parser(commands)
    add_language_parser(commands)
    gui = commands.add_parser('gui', help='Open the native model workbench')
    gui.add_argument('--workspace', default='.', help='Project/data directory (default: current directory)')
    gui.add_argument('--checkpoint', help='Language checkpoint to load at startup')
    args = parser.parse_args()
    try:
        if args.command == 'gui':
            try:
                from .workbench.app import run_gui
            except ImportError as error:
                raise ValueError('Desktop GUI requires Tkinter/Tk. Install the Python Tk package for your OS.') from error
            run_gui(args)
        elif args.command == 'language':
            run_language(args)
        else:
            run_simulation(args)
    except (ValueError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
