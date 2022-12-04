"""Read an Abaqus .inp file for elem/node data"""


import typing
import statistics

FN_INP = r"data/exp5-sd4.inp"
FN_INP_OUT = r"data/exp5-sd4-out-{suffix}.inp"

class XYZ(typing.NamedTuple):
    x: float
    y: float
    z: float


class Element(typing.NamedTuple):
    num: int
    connection: typing.Tuple[int, ...]


def _average_pos(*xyzs) -> XYZ:

    # Rotate to components
    comps = zip(*xyzs)
    averages = [statistics.mean(vals) for vals in comps]
    return XYZ(*averages)

def _get_lines_starting_from(key: str, file_lines):
    in_section_of_interest = False
    finished = False


    for l in file_lines:
        on_header = False
        if l.strip() == key:
            if finished:
                raise ValueError(f"Got a second line matching {key}: {l}")
            
            in_section_of_interest = True
            on_header = True

        elif (in_section_of_interest and l.startswith("*")):
            in_section_of_interest = False
            finished = True

        if in_section_of_interest and not on_header:
            yield l


def get_nodes(file_lines: typing.List[str]) -> typing.Iterable[typing.Tuple[int, XYZ]]:
    for l in _get_lines_starting_from("*Node", file_lines):
        nxyz = l.split(",")
        n = int(nxyz[0])
        xyz = [float(x) for x in nxyz[1:4]]
        # Can drop the z if it's all planar
        if len(xyz) == 2:
            xyz.append(0.0)
        yield n, XYZ(*xyz)

def get_elements(file_lines: typing.List[str]) -> typing.Iterable[Element]:
    for l in _get_lines_starting_from("*Element, type=CPS4", file_lines):
        nconn = [int(x) for x in l.split(',')]
        n = nconn[0]
        conn = tuple(nconn[1:])
        yield Element(n, conn)

def get_element_and_cent_absolute(file_lines) -> typing.Iterable[typing.Tuple[Element, XYZ]]:
    nodes = {n: xyz for n, xyz in get_nodes(file_lines)}
    for elem in get_elements(file_lines):
        these_nodes = [nodes[nn] for nn in elem.connection]
        cent_xyz = _average_pos(*these_nodes)
        yield elem, cent_xyz

def get_element_and_cent_relative(file_lines) -> typing.Iterable[typing.Tuple[Element, XYZ]]:
    """Puts everything on the 0.0...1.0 range"""

    lower, upper = get_bounds(file_lines)

    def normalise(abs_point, idx):
        l = lower[idx]
        u = upper[idx]
        p = abs_point[idx]

        if p < l:
            raise ValueError(f"Query point {p} less than minimum {l} for {idx}??")

        if p > u:
            raise ValueError(f"Query point {p} more than maximum {u} for {idx}??")

        diff = u-l
        if not diff:
            return p

        return (p - l) / diff

        

    for elem, cent_xyz in get_element_and_cent_absolute(file_lines):
        yield elem, XYZ(x=normalise(cent_xyz, 0), y=normalise(cent_xyz, 1), z=normalise(cent_xyz, 2))


def get_bounds(file_lines) -> typing.Tuple[XYZ, XYZ]:
    """Min and max of all the dimensions"""

    node_pos = {xyz for _, xyz in get_nodes(file_lines)}
    components = list(zip(*node_pos))

    lower = [min(xxx) for xxx in components]
    upper = [max(xxx) for xxx in components]
    return XYZ(*lower), XYZ(*upper)


if __name__ == "__main__":
    with open(FN_INP) as f:
        file_lines = f.readlines()

    print(get_bounds(file_lines))

    aaa = {en:xyz for en, xyz in get_element_and_cent_absolute(file_lines)}

    for n, xyz in get_element_and_cent_relative(file_lines):
        print(n, xyz)




