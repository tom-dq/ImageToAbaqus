import collections

import parse_inp
import read_image

def main():
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
        labeled_point = raw_labeled[idx_x, idx_y]
        
        prop_to_elem_nums[labeled_point].append(n.num)

    for grain_num, elems in sorted(prop_to_elem_nums.items()):
        print(grain_num, elems)

    # TODO - up to here. Need to output the section assignemnts!


if __name__ == "__main__":
    main()
    