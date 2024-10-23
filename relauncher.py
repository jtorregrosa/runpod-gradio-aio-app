import os
import argparse
import time

def relaunch_process(command, launch_counter=0):
    '''
    Relaunch a given command indefinitely with a delay between restarts.
    '''
    while True:
        print('Relauncher: Launching...')
        if launch_counter > 0:
            print(f'\tRelaunch count: {launch_counter}')

        try:
            # Execute the command using os.system
            os.system(command)
        except Exception as err:
            print(f"An unexpected error occurred: {err}")
        finally:
            print('Relauncher: Process is ending. Relaunching in 2s...')
            launch_counter += 1
            time.sleep(2)

def main():
    # Initialize the argument parser
    parser = argparse.ArgumentParser(description='Run a command using Python os.system with relaunch capability.')
    # Add the argument for the command
    parser.add_argument('command', type=str, nargs='+', help='The command to execute')
    # Parse the command-line arguments
    args = parser.parse_args()

    # Join the command parts to form the complete command string
    command = ' '.join(args.command)

    # Launch the command with relaunch capability
    relaunch_process(command)

if __name__ == '__main__':
    main()
