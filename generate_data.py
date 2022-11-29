import argparse
import os
import os.path
import numpy as np
import scipy.signal


parser = argparse.ArgumentParser(description='Generate data')
parser.add_argument('output', type=str, nargs='?', default='data',
                    help='Output name')
parser.add_argument('--mask', type=str, default='mask',
                    help='Mask to use when generating data')
parser.add_argument('--num-steps', type=int, default=100,
                    help='Number of steps in each run')
parser.add_argument('--num-runs', type=int, default=100,
                    help='Number of runs to do')
parser.add_argument('--seed', type=int, default=42,
                    help='Random seed')
parser.add_argument('--num-sources', type=int, default=3,
                    help='Max number of "heat" sources in random locations')
parser.add_argument('--neg-sources', default=False, action='store_true',
                    help='Allow initial sources to be negative')
parser.add_argument('--zero-init', default=False, action='store_true',
                    help='Initialize data to 0')
parser.add_argument('--init-ranges', type=int, default=[0, 0.1], nargs='+',
                    help='Ranges for generating initial random data')
parser.add_argument('--clip', type=int, nargs=2, default=None,
                    help='Clip to range')
parser.add_argument('--diffusivity', type=float, nargs='+', default=None,
                    help='Initialize kernels to be heat diffusion')
parser.add_argument('--heat-rate', type=int, default=None,
                    help='Inject additional heat every this many steps')
parser.add_argument('--wrap-domain', default=False, action='store_true',
                    help='Wrap domain boundaries')
args = parser.parse_args()

if not os.path.exists(args.mask + '.npy'):
    raise RuntimeError('No mask')
mask = np.load(args.mask + '.npy')
num_types = len(np.unique(mask))

rng = np.random.default_rng(seed=args.seed)

if args.diffusivity:
    if len(args.diffusivity) != num_types:
        raise ValueError('Wrong number of diffusivities')
    filters = []
    for diffusivity in args.diffusivity:
        if 1. - 4*diffusivity < 0.:
            print(f'Warning: diffusivity {diffusivity} may lead to'
                  ' negative heat')
        filters.append(np.array(
            [[0., diffusivity, 0.],
             [diffusivity, 1. - 4*diffusivity, diffusivity],
             [0., diffusivity, 0.]]
        ))
else:
    raise RuntimeError('No fallback without diffusivity')

if args.neg_sources and args.heat_rate:
    raise RuntimeError('Cannot have --neg-sources and --heat-rate')

if args.wrap_domain:
    conv_boundary = 'wrap'
else:
    conv_boundary = 'fill'


def init_data():
    """Randomly initialize data and return it."""
    data = np.zeros(mask.shape)
    if not args.zero_init:
        if len(args.init_ranges) == 2:
            data = rng.uniform(*args.init_ranges, mask.shape)
        elif len(args.init_ranges) == 2*num_types:
            for t in range(num_types):
                init_range = args.init_ranges[t*2:t*2+2]
                data[mask == t] = rng.uniform(*init_range, mask.shape)[mask == t]
        else:
            raise ValueError('Bad --init-ranges')
    saved_indices = []
    num_choices = rng.integers(1, args.num_sources, endpoint=True)
    for t in range(num_types):
        indices = np.argwhere(mask == t)
        indices = rng.choice(indices, size=num_choices)
        for idx in indices:
            saved_indices.append(tuple(idx))
            if args.neg_sources:
                if rng.integers(0, 1, endpoint=True) == 0:
                    data[tuple(idx)] = 1.0
                else:
                    data[tuple(idx)] = -1.0
            else:
                data[tuple(idx)] = 1.0
    return data, saved_indices


def generate_data(data, heat_indices):
    """Generate steps starting from data (which will be modified)."""
    saved_data = np.empty((args.num_steps + 1,) + mask.shape, dtype=np.float32)
    for i in range(args.num_steps):
        # For now we do the convolution over all data and then only
        # select the relevant portion. Not the most efficient, but simple.
        # We apply the updates at the end so each conv sees the same data.
        saved_data[i, :] = data[:]
        if args.heat_rate and (i + 1) % args.heat_rate == 0 and i > 0:
            for idx in heat_indices:
                data[idx] += 1.0
        updates = []
        for t in range(num_types):
            conv = scipy.signal.convolve2d(
                data, filters[t], mode='same', boundary=conv_boundary)
            if args.clip:
                np.clip(conv, *args.clip, out=conv)
            updates.append(conv)
        for t in range(num_types):
            data[mask == t] = updates[t][mask == t]
    saved_data[-1, :] = data[:]  # Save the last step.
    return saved_data


# Generate data for all runs.
all_data = np.empty((args.num_runs, args.num_steps + 1) + mask.shape,
                    dtype=np.float32)
for run in range(args.num_runs):
    run_data, heat_indices = init_data()
    run_data = generate_data(run_data, heat_indices)
    norm_run = (run_data - run_data.mean()) / run_data.std()
    print(f'Run {run}: min={run_data.min()} max={run_data.max()}'
          f' absmin={np.abs(run_data).min()} absmax={np.abs(run_data).max()}'
          f' stdmin={norm_run.min()} stdmax={norm_run.max()}')
    all_data[run, :] = run_data

# Save data.
np.save(args.output, all_data)
