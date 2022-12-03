import collections
import enum
import itertools
import typing

import parse_inp
import read_image

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
        raise ValueError('n must be at least one')
    it = iter(iterable)
    while (batch := list(itertools.islice(it, n))):
        yield batch


def _put_in_lines(tokens, n_per_line: int) -> typing.Iterable[str]:
    """Chunk things up to some number on each line, Abaqus style."""
    for line_chunk in batched(tokens, n_per_line):
        yield ", ".join(str(x) for x in line_chunk)


def get_grain_labels():
    # Get the labeled image with "grains"
    img = read_image.read_raw_img(read_image.FN)
    _, raw_labeled = read_image.mean_shift(50, 1000, img)

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
        labeled_point = one_based_start
        
        prop_to_elem_nums[labeled_point].append(n.num)

    return prop_to_elem_nums


def get_name(ent_type: AbaInpEnt, label) -> str:
    return f"{ent_type.value}{label}"

def _one_elset(label, elems):
    yield f"*Elset, elset={get_name(AbaInpEnt.elset, label)}"
    yield from _put_in_lines(elems, 16)

def _one_section(label, elems):
    yield f"** Section: {get_name(AbaInpEnt.section, label)}"
    yield f"*Solid Section, elset={get_name(AbaInpEnt.elset, label)}, material={get_name(AbaInpEnt.material, label)}"

def make_abaqus_lines(prop_to_elem_nums: typing.Dict[str, typing.List[int]]) -> typing.Iterable[str]:
    # Element sets
    
    # Make an "all elements" set for enrichemnt etc
    all_elems = set()
    for elems in prop_to_elem_nums.values():
        all_elems.update(elems)

    # yield from _one_elset("All", sorted(all_elems))

    for maker_func in [_one_elset, _one_section]:
        for label, elems in sorted(prop_to_elem_nums.items()):
            yield from maker_func(label, elems)


    


    """** Section: crystal-50  
*Solid Section, elset=grain-50, material=metal-50  
,  """



if __name__ == "__main__":
    prop_to_elem_nums = get_grain_labels()
    for l in make_abaqus_lines(prop_to_elem_nums):
        print(l)

