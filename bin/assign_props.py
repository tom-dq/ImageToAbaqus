import collections
import enum
import random
import itertools
import typing
import pathlib
import functools

import grain_partition
import parse_inp
import read_image


class MatSource(enum.Enum):
    general = "general"
    mao_paper = "mao_paper"


class RunOptions(typing.NamedTuple):
    material_source: MatSource
    mean_shift_bandwidth: int  # Higher bandwidth -> fewer distinct grains
    random_seed: int
    critical_contiguous_grain_ratio: float


class AbaInpEnt(enum.Enum):
    # Key is entity type. Value is prefix of name
    elset = "grain-"
    section = "crystal-"
    material = "metal-"


def batched(iterable, n):
    "Batch data into lists of length n. The last batch may be shorter."
    # https://docs.python.org/3/library/itertools.html
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError("n must be at least one")
    it = iter(iterable)
    while batch := list(itertools.islice(it, n)):
        yield batch


def _put_in_lines(tokens, n_per_line: int) -> typing.Iterable[str]:
    """Chunk things up to some number on each line, Abaqus style."""
    for line_chunk in batched(tokens, n_per_line):
        yield ", ".join(str(x) for x in line_chunk)


@functools.lru_cache(maxsize=256)
def get_grain_labels(
    run_options: RunOptions,
) -> typing.Dict[str, typing.List[parse_inp.Element]]:
    # Get the labeled image with "grains"
    img = read_image.read_raw_img(read_image.FN)
    _, raw_labeled = read_image.mean_shift(run_options.mean_shift_bandwidth, 1000, img)

    # Read in the INP
    with open(parse_inp.FN_INP) as f:
        file_lines = f.readlines()

    NX, NY = raw_labeled.shape

    prop_to_elem_nums = collections.defaultdict(list)

    for n, xyz in parse_inp.get_element_and_cent_relative(file_lines):
        # Get the nearest labeled point in the
        idx_x = int(xyz.x * NX)
        idx_y = int(xyz.y * NY)

        one_based_start = raw_labeled[idx_x, idx_y] + 1
        labeled_point = str(one_based_start)

        prop_to_elem_nums[labeled_point].append(n)

    return prop_to_elem_nums


class RemoveSmallRegionsWorkingData(typing.NamedTuple):
    region_ratio: float
    cluster: grain_partition.Cluster
    adj_grains: typing.List[typing.Tuple[grain_partition.Cluster, int]]


def _remove_small_grain_regions(
    grain_ratio: float,
    prop_to_elems: typing.Dict[str, typing.List[parse_inp.Element]],
) -> typing.Tuple[int, int, typing.Dict[str, typing.List[parse_inp.Element]]]:

    # Step 1 - Build the adjaceny graph of grains
    cluster_set = grain_partition.ClusterSet()
    labeled_elems = []
    for label, fe_elems in prop_to_elems.items():
        for fe_elem in fe_elems:
            labeled_elems.append(
                grain_partition.LabeledElement(
                    label=str(label), num=fe_elem.num, connection=fe_elem.connection
                )
            )

    cluster_set.add_all_elems(labeled_elems)

    # Step 2 - organise the clusters by size.
    with open(parse_inp.FN_INP) as f:
        elem_size = parse_inp.get_areas(f.readlines())

    total_area = sum(elem_size.values())

    def produce_clusters_and_sizes():
        adj_func = cluster_set.produce_adjacency_function()

        for cluster in cluster_set.clusters.values():
            elem_nums = {e.num for e in cluster.elements}
            region_ratio = sum(elem_size[e] for e in elem_nums) / total_area
            yield RemoveSmallRegionsWorkingData(
                region_ratio=region_ratio, cluster=cluster, adj_grains=adj_func(cluster)
            )

    working_list = list(produce_clusters_and_sizes())
    working_list.sort()

    # Set this up, then overwrite it with any changes needed.
    elem_num_to_prop = dict()
    elem_num_to_fe_elem = dict()
    for label, fe_elems in prop_to_elems.items():
        for fe_elem in fe_elems:
            elem_num_to_prop[fe_elem.num] = label
            elem_num_to_fe_elem[fe_elem.num] = fe_elem

    # Go through the list, smallest to largest. If it's too small, switch for something
    # with which many edges are shared, and which is bigger.
    removed_cluster_ids = set()
    clusters_removed = 0
    element_changed = 0
    for working_data in working_list:
        to_remove = working_data.region_ratio < grain_ratio
        if to_remove:
            # Find a candidate which has not itself been removed.
            valid_candidates = [
                (common_edges, cluster)
                for cluster, common_edges in working_data.adj_grains
                if id(cluster) not in removed_cluster_ids
            ]

            # Find the one with the most shared edges.
            valid_candidates.sort(reverse=True)

            if not valid_candidates:
                # Sometimes all of a regions's neighbors have gone already. Have to wait
                # until the next iteration of this in that case!
                continue

            migrate_to = valid_candidates[0][1]
            for labeled_elem in working_data.cluster.elements:
                elem_num_to_prop[labeled_elem.num] = migrate_to.label

            clusters_removed += 1
            element_changed += len(working_data.cluster.elements)
            removed_cluster_ids.add(id(working_data.cluster))

    # Build the output data structure in the same way as we provided it.
    out_prop_to_elems = collections.defaultdict(list)
    for elem_num, label in elem_num_to_prop.items():
        out_prop_to_elems[label].append(elem_num_to_fe_elem[elem_num])

    return clusters_removed, element_changed, out_prop_to_elems


def get_name(ent_type: AbaInpEnt, label) -> str:
    return f"{ent_type.value}{label}"


def _one_elset(label, elems):
    yield f"*Elset, elset={get_name(AbaInpEnt.elset, label)}"
    yield from _put_in_lines(elems, 16)


def _one_section(label, elems):
    yield f"** Section: {get_name(AbaInpEnt.section, label)}"
    yield f"*Solid Section, elset={get_name(AbaInpEnt.elset, label)}, material={get_name(AbaInpEnt.material, label)}"


def make_abaqus_lines(
    prop_to_elem_nums: typing.Dict[str, typing.List[int]]
) -> typing.Iterable[str]:
    # Element sets

    # Make an "all elements" set for enrichemnt etc
    all_elems = set()
    for elems in prop_to_elem_nums.values():
        all_elems.update(elems)

    # yield from _one_elset("All", sorted(all_elems))

    for maker_func in [_one_elset, _one_section]:
        for label, elems in sorted(prop_to_elem_nums.items()):
            debug_check = 10254 in elems
            if debug_check:
                # print(label, elems)
                pass

            yield from maker_func(label, elems)


def get_material_reference_lines(
    mat_source: MatSource, material_names: typing.Iterable[str]
) -> typing.Iterable[str]:
    """Gets some number of material def lines, straight out of a reference file."""

    match mat_source:
        case MatSource.general:
            fn = "data/mat_orig.txt"

        case MatSource.mao_paper:
            fn = "data/mat_mao.txt"

        case _:
            raise ValueError(mat_source)

    line_chunks = []

    # List of lists of material data

    working_set = list()

    with open(fn) as f:
        for l in (l.strip() for l in f if l.strip()):
            is_starting_line = l.startswith("*Material, name=")
            if is_starting_line:
                if working_set:
                    line_chunks.append(working_set)

                working_set = list()

            # This bit always happens, even for a new group of lines.
            working_set.append(l)

    # Now, get a random few of them
    random.shuffle(line_chunks)

    names_to_return = collections.deque(material_names)
    while names_to_return:
        this_name = names_to_return.popleft()

        one_from_file = line_chunks.pop()

        # Set the name to the right thing
        if not one_from_file[0].startswith("*Material, name="):
            raise ValueError(one_from_file[0])

        one_from_file[0] = f"*Material, name={this_name}"

        yield from one_from_file


def iterate_until_no_small_grains(
    run_options: RunOptions,
    prop_to_elems: typing.Dict[str, typing.List[parse_inp.Element]],
) -> typing.Dict[str, typing.List[parse_inp.Element]]:

    if not run_options.critical_contiguous_grain_ratio:
        print(f"Skipping small grain removal.")
        return prop_to_elems

    num_grains, num_elem = 1, 1
    while num_grains or num_elem:
        num_grains, num_elem, prop_to_elems = _remove_small_grain_regions(
            run_options.critical_contiguous_grain_ratio, prop_to_elems
        )

        print(
            f"Removed {num_grains} small grains by adjusting the property of {num_elem} elements."
        )

    return prop_to_elems


class InsertPos(enum.Enum):
    before = "before"
    after = "after"


@functools.lru_cache(maxsize=256)
def get_grain_labels_and_remove_small_grains(run_options: RunOptions):
    prop_to_elems = get_grain_labels(run_options)

    # Get rid of the small grains
    prop_to_elems = iterate_until_no_small_grains(run_options, prop_to_elems)

    return prop_to_elems


def interleave(run_options: RunOptions, source_lines) -> typing.Iterable[str]:
    """Mix in the old and the new"""

    # Make this so we can cache the label output
    cacheable_run_options = run_options._replace(random_seed=None)

    prop_to_elems = get_grain_labels_and_remove_small_grains(cacheable_run_options)

    prop_to_elem_nums = {}
    for label, elems in prop_to_elems.items():
        prop_to_elem_nums[label] = [e.num for e in elems]

    el_section_lines = list(make_abaqus_lines(prop_to_elem_nums))

    material_names = [
        get_name(AbaInpEnt.material, label) for label in prop_to_elem_nums.keys()
    ]
    material_prop_lines = list(
        get_material_reference_lines(run_options.material_source, material_names)
    )

    insertions = {
        "*End Part": (
            InsertPos.before,
            el_section_lines,
        ),  # Element sections assignments grain by grain
        "*End Assembly": (InsertPos.after, material_prop_lines),
    }

    for l in source_lines:
        lines_this_chunk = [l]
        maybe_insertion = insertions.get(l.strip(), None)
        if maybe_insertion:
            pos, data = maybe_insertion
            if pos == InsertPos.before:
                lines_this_chunk = data + [l]

            elif pos == InsertPos.after:
                lines_this_chunk.extend(data)

        yield from lines_this_chunk


def make_out_file(run_options: RunOptions):
    orig_path = pathlib.Path(parse_inp.FN_INP)

    grain_format = (  # No decimals for Abaqus!
        f"{run_options.critical_contiguous_grain_ratio:1.2e}"
        if run_options.critical_contiguous_grain_ratio
        else "None"
    )
    grain_format = grain_format.replace(".", "o")
    name_with_suffix = (
        orig_path.stem
        + "-"
        + run_options.material_source.name
        + "-B"
        + str(run_options.mean_shift_bandwidth)
        + "-GR"
        + grain_format
        + "-S"
        + str(run_options.random_seed)
    )
    out_fn = pathlib.Path("out") / (name_with_suffix + orig_path.suffix)
    return out_fn


def make_file(run_options: RunOptions):

    # Read in the thing output by Abaqus
    with open(parse_inp.FN_INP) as f:
        file_lines = f.readlines()

    random.seed(run_options.random_seed)

    fn_out = make_out_file(run_options)

    with open(fn_out, "w") as f_out:
        for l in interleave(run_options, file_lines):
            if not l.endswith("\n"):
                l = l + "\n"
            f_out.write(l)

    print(str(fn_out))


if __name__ == "__main__":

    for critical_contiguous_grain_ratio in [
        # None,
        # 0.00001,
        # 0.000025,
        # 0.00005,
        # 0.0001,  # Looks good?
        0.00025,
        0.001,
        # 0.0025,
    ]:
        # for bw in [30, 40, 50, 60]:
        for bw in [
            25,
        ]:
            for random_seed in range(3000, 3200):
                run_options = RunOptions(
                    material_source=MatSource.mao_paper,
                    mean_shift_bandwidth=bw,
                    random_seed=random_seed,
                    critical_contiguous_grain_ratio=critical_contiguous_grain_ratio,  # 0.00025,
                )

                make_file(run_options)

if False:

    print()
    print()

    prop_to_elem_nums = get_grain_labels()
    for l in make_abaqus_lines(prop_to_elem_nums):
        print(l)
