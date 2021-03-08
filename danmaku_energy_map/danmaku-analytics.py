import argparse
import json
import numpy as np
import matplotlib.pyplot as plt

from danmaku_energy_map import read_danmaku_file, convert_time, segment_text, get_heat_time, draw_he_line, \
    draw_he_annotate

parser = argparse.ArgumentParser(description='Get gift analytics for BiliBili Live XML')
parser.add_argument('danmaku', type=str, help='path to the danmaku file')
parser.add_argument('--graph', type=str, default=None, help='output graph path, leave empty if not needed')
# parser.add_argument('--he_map', type=str, default=None, help='output high density timestamp, leave empty if not needed')
# parser.add_argument('--sc_list', type=str, default=None, help='output super chats, leave empty if not needed')


if __name__ == '__main__':
    args = parser.parse_args()
    xml_list = read_danmaku_file(args.danmaku)

    if args.graph is not None:
        heat_time, heat_value_gaussian, heat_value_gaussian2, he_points = get_heat_time(xml_list)

        fig = plt.figure(figsize=(heat_time[0][-1] / 1000, 4))

        draw_he_line(fig.gca(), heat_time, heat_value_gaussian, heat_value_gaussian2)

        for name, name_chn in {'d': "danmaku", 'sc': "superchat", 'gift': "gift"}.items():
            part_xml_list = [element for element in xml_list if element.tag == name]
            heat_time, heat_value_gaussian, heat_value_gaussian2, _ = get_heat_time(part_xml_list)
            if name in ['sc', 'gift']:
                draw_he_line(fig.gca(), heat_time, heat_value_gaussian * 10, heat_value_gaussian2 * 10, name=name_chn)

        part_xml_list = [element for element in xml_list if element.tag in ['sc', 'gift']]
        heat_time, heat_value_gaussian, heat_value_gaussian2, he_points = get_heat_time(part_xml_list)
        draw_he_annotate(fig.gca(), heat_time, he_points)

        t_x = heat_time[0][::1000]

        t = [convert_time(time) for time in t_x]

        plt.xticks(t_x, t)

        plt.legend()

        plt.savefig(args.graph)
