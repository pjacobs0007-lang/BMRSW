import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--p_bounds', type=float, nargs='+', required=True)
args = parser.parse_args(['--p_bounds', '0', '5', '0', '2'])
print(args.p_bounds)
