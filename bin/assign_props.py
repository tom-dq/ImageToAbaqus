import collections
import enum
import random
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
            debug_check = 10254 in elems
            if debug_check:
                # print(label, elems)
                pass

            yield from maker_func(label, elems)


def get_material_reference_lines(material_names: typing.Iterable[str]) -> typing.Iterable[str]:
    """Gets some number of material def lines, straight out of a reference file."""

    line_chunks = []

    # List of lists of material data

    working_set = list()

    with open(r"data/example_materials.txt") as f:
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


class InsertPos(enum.Enum):
    before = "before"
    after = "after"

def interleave(source_lines) -> typing.Iterable[str]:
    """Mix in the old and the new"""

    prop_to_elem_nums = get_grain_labels()
    el_section_lines = list(make_abaqus_lines(prop_to_elem_nums))

    material_names = [get_name(AbaInpEnt.material, label) for label in prop_to_elem_nums.keys()]
    material_prop_lines = list(get_material_reference_lines(material_names))

    insertions = {
        "*End Part": (InsertPos.before, el_section_lines),  # Element sections assignments grain by grain
        "*End Assembly": (InsertPos.after, material_prop_lines)
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
    

    

def make_file():

    run = 12

    # Read in the thing output by Abaqus
    with open(parse_inp.FN_INP) as f:
        file_lines = f.readlines()

    random.seed(run)
    with open(parse_inp.FN_INP_OUT.format(suffix=run), 'w') as f_out:
        for l in interleave(file_lines):
            if not l.endswith('\n'):
                l = l + "\n"
            f_out.write(l)


    """** Section: crystal-50  
*Solid Section, elset=grain-50, material=metal-50  
,  """



if __name__ == "__main__":
    
        
    make_file()

if False:

    print()
    print()


    prop_to_elem_nums = get_grain_labels()
    for l in make_abaqus_lines(prop_to_elem_nums):
        print(l)


