import argparse
import json
import xml.etree.ElementTree as ET
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.filters import gaussian_filter

parser = argparse.ArgumentParser(description='Process bilibili Danmaku')
parser.add_argument('danmaku', type=str, help='path to the danmaku file')
parser.add_argument('--graph', type=str, default=None, help='output graph path, leave empty if not needed')
parser.add_argument('--he_map', type=str, default=None, help='output high density timestamp, leave empty if not needed')
parser.add_argument('--sc_list', type=str, default=None, help='output super chats, leave empty if not needed')


def read_danmaku_file(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()

    all_children = [child for child in root if child.tag in ['gift', 'sc', 'd']]
    return all_children


def get_time(child: ET.Element):
    # noinspection PyBroadException
    try:
        if child.tag == 'd':
            return float(child.attrib['p'].split(',')[0])
        elif child.tag == 'gift' or child.tag == 'sc':
            return float(child.attrib['ts'])
    except:
        print(f"error getting time from {child}")
        return 0


def get_value(child: ET.Element):
    # noinspection PyBroadException
    try:
        if child.tag == 'd':
            return 1
        elif child.tag == 'gift':
            raw_data = json.loads(child.attrib['raw'])
            return raw_data['total_coin'] / 1000 / 10
        elif child.tag == 'sc':
            raw_data = json.loads(child.attrib['raw'])
            return raw_data['price'] / 10
    except:
        print(f"error getting value from {child}")
        return 0


def get_heat_time(all_children):
    interval = 2

    center = 0

    cur_entry = 0

    final_time = get_time(all_children[-1])

    cur_heat = 0

    danmaku_queue = deque()

    heat_time = [[], []]

    while True:
        if center > final_time:
            break

        start = center - interval
        end = center + interval

        while cur_entry < len(all_children) and get_time(all_children[cur_entry]) < end:
            cur_danmaku = all_children[cur_entry]
            danmaku_queue.append(cur_danmaku)
            cur_heat += get_value(cur_danmaku)
            cur_entry += 1

        while len(danmaku_queue) != 0 and get_time(danmaku_queue[0]) < start:
            prev_danmaku = danmaku_queue.popleft()
            cur_heat -= get_value(prev_danmaku)

        heat_time[0] += [center]
        heat_time[1] += [cur_heat]
        center += 1

    heat_value = heat_time[1]
    heat_value_gaussian = gaussian_filter(heat_value, sigma=50)
    heat_value_gaussian2 = gaussian_filter(heat_value, sigma=1000) * 1.1

    he_points = [[], []]
    cur_highest = -1
    highest_idx = -1

    for i in range(len(heat_value_gaussian)):
        if highest_idx != -1:
            if heat_value_gaussian[i] < heat_value_gaussian2[i]:
                he_points[0] += [highest_idx]
                he_points[1] += [cur_highest]
                highest_idx = -1
            else:
                if heat_value_gaussian[i] > cur_highest:
                    cur_highest = heat_value_gaussian[i]
                    highest_idx = i
        else:
            if heat_value_gaussian[i] > heat_value_gaussian2[i]:
                cur_highest = heat_value_gaussian[i]
                highest_idx = i

    # Usually the HE point at the end of a live stream is just to say goodbye
    # if highest_idx != -1:
    #     he_points[0] += [highest_idx]
    #     he_points[1] += [cur_highest]

    return heat_time, heat_value_gaussian, heat_value_gaussian2, he_points


def convert_time(secs):
    minutes = secs // 60
    reminder_secs = secs % 60
    return f"{minutes}:{reminder_secs:02d}"


def draw_he_line(fig: plt.Figure, heat_time, heat_value_gaussian, heat_value_gaussian2, name='all', no_average=False):
    fig.gca().plot(heat_time[0], heat_value_gaussian, label=f"{name}")
    if not no_average:
        fig.gca().plot(heat_time[0], heat_value_gaussian2, label=f"{name} average")


def draw_he_annonate(fig: plt.Figure, heat_time, he_points):
    for i in range(len(he_points[0])):
        time = heat_time[0][he_points[0][i]] - 45
        time_name = convert_time(time)
        height = he_points[1][i]
        fig.gca().annotate(time_name, xy=(time, height), xytext=(time, height + 5),
                           arrowprops=dict(facecolor='black', shrink=0.05))


def draw_he(he_graph, heat_time, heat_value_gaussian, heat_value_gaussian2, he_points):
    fig = plt.figure(figsize=(heat_time[0][-1] / 1000, 4))

    draw_he_line(fig, heat_time, heat_value_gaussian, heat_value_gaussian2)
    draw_he_annonate(fig, heat_time, he_points)

    t_x = heat_time[0][::1000]

    t = [convert_time(time) for time in t_x]

    plt.xticks(t_x, t)

    plt.savefig(he_graph)


TEXT_LIMIT = 900
SEG_CHAR = '\n\n\n\n'


def segment_text(text):
    lines = text.split('\n')
    new_text = ""
    new_segment = ""

    for line in lines:
        if len(new_segment) + len(line) < TEXT_LIMIT:
            new_segment += line + "\n"
        else:
            if len(line) > TEXT_LIMIT:
                print(f"line\"{line}\" too long, omit.")
            else:
                new_text += new_segment + SEG_CHAR
                new_segment = line + "\n"
    new_text += new_segment
    return new_text


if __name__ == '__main__':
    args = parser.parse_args()
    xml_list = read_danmaku_file(args.danmaku)

    if args.sc_list is not None:
        sc_chats = [element for element in xml_list if element.tag == 'sc']

        sc_tuple = []
        for sc_chat_element in sc_chats:
            try:
                price = sc_chat_element.attrib['price']
                raw_message = json.loads(sc_chat_element.attrib['raw'])
                message = raw_message["message"].replace('\n', '\t')
                time = int(float(sc_chat_element.attrib['ts']))
                sc_tuple += [(time, price, message)]
            except:
                print(f"superchat processing error {sc_chat_element}")

        sc_text = "醒目留言列表："
        for time, price, message in sc_tuple:
            sc_text += f"\n {convert_time(time)} ¥{price}: {message}"
        sc_text += "\n"
        sc_text = segment_text(sc_text)
        with open(args.sc_list, "w") as file:
            file.write(sc_text)

    if args.he_map is not None or args.graph is not None:
        heat_values = get_heat_time(xml_list)

        if args.he_map is not None:
            he_pairs = heat_values[3]
            all_timestamps = heat_values[0][0]
            if len(he_pairs[0]) == 0:
                text = "没有高能..."
            else:
                # noinspection PyTypeChecker
                highest_time_id = he_pairs[0][np.argmax(he_pairs[1])]
                highest_time = all_timestamps[highest_time_id]

                other_he_time_list = [all_timestamps[time_id] for time_id in he_pairs[0]]

                text = f"全场最高能：{convert_time(highest_time - 45)}\n\n其他高能："

                for other_he_time in other_he_time_list:
                    text += f"\n {convert_time(other_he_time - 45)}"
            text += "\n"
            text = segment_text(text)
            with open(args.he_map, "w") as file:
                file.write(text)

        if args.graph is not None:
            draw_he(args.graph, *heat_values)
