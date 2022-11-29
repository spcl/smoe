import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns


sns.set()

parser = argparse.ArgumentParser(description='Generate data mask')
parser.add_argument('output', type=str, nargs='?', default='mask',
                    help='Output name')
parser.add_argument('--shape', type=int, nargs=2, default=(64, 64),
                    help='Shape of the data')
parser.add_argument('--types', type=int, default=3,
                    help='Number of mask types')
parser.add_argument('--plot', action='store_true', default=False,
                    help='Plot the mask')
args = parser.parse_args()

# Dimensions of the mask data.
MASK_SHAPE = args.shape
# Number of different types in the mask.
MASK_TYPES = args.types
# Output name for the saved mask.
MASK_OUTPUT = args.output
# Whether to save a plot of the mask.
MASK_PLOT = args.plot

# Starting type probabilities.
TYPE_PROBS = np.full(MASK_TYPES, 1.0 / MASK_TYPES)
# Amount to increment a type's probability by when there is a neighbor
# of that type.
# We need to divide (1 - 1/MASK_TYPES) by at least 8 to make the
# probabilities nice.
# While using 8 means that if all neighbors are the same, the location
# will get that type, most of the time we won't have all neighbors
# chosen.
TYPE_INCR = (1.0 - 1.0 / MASK_TYPES) / 5
# Amount to decrease other type's probabilities by.
TYPE_DECR = TYPE_INCR / (MASK_TYPES - 1) if MASK_TYPES > 1 else 0.

rng = np.random.default_rng(seed=42)


def get_type(neighborhood):
    """Return the new type for a location given its neighborhood.

    Locations with no type (which can include the location for which
    the type is desired) should have value -1.

    """
    u, counts = np.unique(neighborhood, return_counts=True)
    probs = TYPE_PROBS.copy()
    # Drop the -1s.
    for t, cnt in zip(u[1:], counts[1:]):
        probs[t] += TYPE_INCR * cnt
        probs[np.arange(MASK_TYPES) != t] -= TYPE_DECR * cnt
    return rng.choice(MASK_TYPES, p=probs)


def square_borders(x, y):
    return {((x, y), (x + 1, y)),
            ((x, y + 1), (x + 1, y + 1)),
            ((x, y), (x, y + 1)),
            ((x + 1, y), (x + 1, y + 1))}


def merge_segments(segments, t, mask):
    horiz_segs = []
    vert_segs = []
    # Find horizontal and vertical segments and contract them toward
    # their type.
    for seg in segments:
        ((x0, y0), (x1, y1)) = seg
        if x0 == x1:
            if x0 == 0:
                # Contract right.
                vert_segs.append(((x0 + 0.1, y0 - 0.1), (x1 + 0.1, y1 - 0.1)))
            elif x0 == mask.shape[1] - 1:
                # Contract left.
                vert_segs.append(((x0 - 0.1, y0 - 0.1), (x1 - 0.1, y1 - 0.1)))
            elif mask[y0, x0 - 1] == t:
                # Contract left.
                vert_segs.append(((x0 - 0.1, y0 - 0.1), (x1 - 0.1, y1 - 0.1)))
            elif mask[y0, x0] == t:
                # Contract right.
                vert_segs.append(((x0 + 0.1, y0 - 0.1), (x1 + 0.1, y1 - 0.1)))
            else:
                print('Could not figure out how to contract')
            vert_segs.append(seg)
        else:
            if y0 == 0:
                # Contract down.
                horiz_segs.append(((x0 - 0.1, y0 + 0.1), (x1 - 0.1, y1 + 0.1)))
            elif y0 == mask.shape[0] - 1:
                # Contract up.
                horiz_segs.append(((x0 - 0.1, y0 - 0.1), (x1 - 0.1, y1 - 0.1)))
            elif mask[y0 - 1, x0] == t:
                # Contract up.
                horiz_segs.append(((x0 - 0.1, y0 - 0.1), (x1 - 0.1, y1 - 0.1)))
            elif mask[y0, x0] == t:
                # Contract down.
                horiz_segs.append(((x0 - 0.1, y0 + 0.1), (x1 - 0.1, y1 + 0.1)))
            else:
                print('Could not figure out how to contract')
            horiz_segs.append(seg)
    # Merge adjacent horizontal segments.
    horiz_segs = sorted(horiz_segs, key=lambda x: (x[0][1], x[0][0]))
    merged_horiz_segs = []
    seg_to_extend = horiz_segs[0]
    for seg in horiz_segs[1:]:
        # Check whether we can extend.
        if seg_to_extend[1] == seg[0]:
            seg_to_extend = (seg_to_extend[0], seg[1])
        else:
            merged_horiz_segs.append(seg_to_extend)
            seg_to_extend = seg
    merged_horiz_segs.append(seg_to_extend)
    # Merge adjacent vertical segments.
    vert_segs = sorted(vert_segs, key=lambda x: x[0])
    merged_vert_segs = []
    seg_to_extend = vert_segs[0]
    for seg in vert_segs[1:]:
        if seg_to_extend[1] == seg[0]:
            seg_to_extend = (seg_to_extend[0], seg[1])
        else:
            merged_vert_segs.append(seg_to_extend)
            seg_to_extend = seg
    merged_vert_segs.append(seg_to_extend)
    return merged_horiz_segs + merged_vert_segs


def get_mask_borders(mask):
    type_borders = [set() for i in range(MASK_TYPES)]
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            t = mask[i, j]
            borders = square_borders(j, i)
            type_borders[t] ^= borders
    return [merge_segments(s, t, mask) for t, s in enumerate(type_borders)]


# Generate the mask.
mask = np.full(MASK_SHAPE, -1)
for i in range(MASK_SHAPE[0]):
    for j in range(MASK_SHAPE[1]):
        neighborhood = mask[max(0, i-1):min(MASK_SHAPE[0], i+2),
                            max(0, j-1):min(MASK_SHAPE[1], j+2)]
        mask[i, j] = get_type(neighborhood)

# Save the mask.
np.save(MASK_OUTPUT, mask)

# Plot the mask.
if MASK_PLOT:
    fig, ax = plt.subplots(1, 1)
    colors = [sns.color_palette()[i] for i in range(MASK_TYPES)]
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        'Custom', colors, len(colors))
    sns.heatmap(mask, cmap=cmap, cbar=False, ax=ax, alpha=0.75)
    plt.setp(plt.yticks()[1], rotation=0)
    type_borders = get_mask_borders(mask)
    for t in range(MASK_TYPES):
        ax.add_collection(matplotlib.collections.LineCollection(
            list(type_borders[t]), colors=colors[t], linewidths=1
        ))
    fig.tight_layout()
    fig.savefig(MASK_OUTPUT + '.pdf')
