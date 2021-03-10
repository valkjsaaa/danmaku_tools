import argparse
import json
import xml.etree.ElementTree as ET
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.filters import convolve
from scipy.stats import halfnorm

parser = argparse.ArgumentParser(description='Process bilibili Danmaku')
parser.add_argument('danmaku', type=str, help='path to the danmaku file')
parser.add_argument('--graph', type=str, default=None, help='output graph path, leave empty if not needed')
parser.add_argument('--he_map', type=str, default=None, help='output high density timestamp, leave empty if not needed')
parser.add_argument('--sc_list', type=str, default=None, help='output super chats, leave empty if not needed')
parser.add_argument('--he_time', type=str, default=None, help='output highest density timestamp, leave empty if not '
                                                              'needed')


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


def half_gaussian_filter(value, sigma):
    space = np.linspace(-4, 4, sigma * 8)
    neg_space = np.linspace(4 * 10, -4 * 10, sigma * 8)
    kernel = (halfnorm.pdf(space) + halfnorm.pdf(neg_space)) / sigma
    offset_value = np.concatenate((value, np.zeros(45)))
    return convolve(offset_value, kernel)[45:]


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
    heat_value_gaussian = half_gaussian_filter(heat_value, sigma=50)
    heat_value_gaussian2 = half_gaussian_filter(heat_value, sigma=1000) * 1.2

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

    return heat_time, heat_value_gaussian / np.sqrt(heat_value_gaussian2), np.sqrt(heat_value_gaussian2), he_points


def convert_time(secs):
    minutes = secs // 60
    reminder_secs = secs % 60
    return f"{minutes}:{reminder_secs:02d}"


def draw_he_line(ax: plt.Axes, heat_time, heat_value_gaussian, heat_value_gaussian2, name='all', no_average=False):
    ax.plot(heat_time[0], heat_value_gaussian, label=f"{name}")
    if not no_average:
        ax.plot(heat_time[0], heat_value_gaussian2, label=f"{name} average")


def draw_he_area(ax: plt.Axes, current_time: float, heat_time, heat_value_gaussian, heat_value_gaussian2, name='all',
                 no_average=False):
    total_len = len(heat_time[0])
    change_pos_list = [0]
    low_area_begin_pos_list = []
    high_area_begin_pos_list = []
    if heat_value_gaussian[0] - heat_value_gaussian2[0] < 0:
        low_area_begin_pos_list.append(0)
    else:
        high_area_begin_pos_list.append(0)
    for i in range(1, total_len - 1):
        prev_diff = heat_value_gaussian[i - 1] - heat_value_gaussian2[i - 1]
        after_diff = heat_value_gaussian[i + 1] - heat_value_gaussian2[i + 1]
        if prev_diff * after_diff < 0:
            change_pos_list.append(i)
            if prev_diff < 0:
                high_area_begin_pos_list.append(len(change_pos_list) - 1)
            else:
                low_area_begin_pos_list.append(len(change_pos_list) - 1)
    change_pos_list.append(total_len - 1)

    for begin_pos_i in low_area_begin_pos_list:
        begin_pos = change_pos_list[begin_pos_i]
        end_pos = change_pos_list[begin_pos_i + 1]
        if end_pos < total_len - 1:
            end_pos += 1
        if current_time <= begin_pos:
            ax.fill_between(heat_time[0][begin_pos:end_pos], heat_value_gaussian[begin_pos:end_pos], color="#f0e442c0",
                            edgecolor="none")
        elif current_time > end_pos:
            ax.fill_between(heat_time[0][begin_pos:end_pos], heat_value_gaussian[begin_pos:end_pos], color="#999999c0",
                            edgecolor="none")
        else:
            ax.fill_between(heat_time[0][begin_pos:current_time], heat_value_gaussian[begin_pos:current_time],
                            color="#999999c0", edgecolor="none")
            ax.fill_between(heat_time[0][current_time:end_pos], heat_value_gaussian[current_time:end_pos],
                            color="#f0e442c0", edgecolor="none")
    if not no_average:
        for begin_pos_i in high_area_begin_pos_list:
            begin_pos = change_pos_list[begin_pos_i]
            end_pos = change_pos_list[begin_pos_i + 1]
            if end_pos < total_len - 1:
                end_pos += 1
            if current_time <= begin_pos:
                ax.fill_between(heat_time[0][begin_pos:end_pos], heat_value_gaussian[begin_pos:end_pos],
                                color="#e69f00c0", edgecolor="none")
            elif current_time > end_pos:
                ax.fill_between(heat_time[0][begin_pos:end_pos], heat_value_gaussian[begin_pos:end_pos],
                                color="#737373c0", edgecolor="none")
            else:
                ax.fill_between(heat_time[0][begin_pos:current_time], heat_value_gaussian[begin_pos:current_time],
                                color="#737373c0", edgecolor="none")
                ax.fill_between(heat_time[0][current_time:end_pos], heat_value_gaussian[current_time:end_pos],
                                color="#e69f00c0", edgecolor="none")


def draw_he_annotate(ax: plt.Axes, heat_time, he_points):
    for i in range(len(he_points[0])):
        time = heat_time[0][he_points[0][i]]
        time_name = convert_time(time)
        height = he_points[1][i]
        ax.annotate(time_name, xy=(time, height), xytext=(time, height + 5),
                    arrowprops=dict(facecolor='black', shrink=0.05))


def draw_he_annotate_line(ax: plt.Axes, current_time: float, heat_time, he_points):
    for i in range(len(he_points[0])):
        time = heat_time[0][he_points[0][i]]
        height = he_points[1][i]
        ax.axline((time, height), (time, height - 1), color='#cc79a7c0')


def draw_he(he_graph, heat_time, heat_value_gaussian, heat_value_gaussian2, he_points, current_time=-1):
    fig = plt.figure(figsize=(16, 1), frameon=False, dpi=60)
    ax = fig.add_axes((0, 0, 1, 1))
    draw_he_area(ax, current_time, heat_time, heat_value_gaussian, heat_value_gaussian2)
    # draw_he_annotate_line(ax, current_time, heat_time, he_points)
    plt.xlim(heat_time[0][0], heat_time[0][-1])
    plt.ylim(min(heat_value_gaussian), max(heat_value_gaussian))

    plt.box(False)
    plt.savefig(he_graph, transparent=True)


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
                user = raw_message["user_info"]['uname']
                time = int(float(sc_chat_element.attrib['ts']))
                sc_tuple += [(time, price, message, user)]
            except:
                print(f"superchat processing error {sc_chat_element}")

        sc_text = "醒目留言列表："
        for time, price, message, user in sc_tuple:
            sc_text += f"\n {convert_time(time)} ¥{price} {user}: {message}"
        sc_text += "\n"
        sc_text = segment_text(sc_text)
        with open(args.sc_list, "w") as file:
            file.write(sc_text)

    if args.he_map is not None or args.graph is not None or args.he_time is not None:
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

                text = f"全场最高能：{convert_time(highest_time)}\n\n其他高能："

                for other_he_time in other_he_time_list:
                    text += f"\n {convert_time(other_he_time)}"
            text += "\n"
            text = segment_text(text)
            with open(args.he_map, "w") as file:
                file.write(text)

        if args.he_time is not None:
            he_pairs = heat_values[3]
            all_timestamps = heat_values[0][0]
            if len(he_pairs[0]) == 0:
                text = "0:00"
            else:
                # noinspection PyTypeChecker
                highest_time_id = he_pairs[0][np.argmax(he_pairs[1])]
                highest_time = all_timestamps[highest_time_id]
                text = convert_time(highest_time)
            with open(args.he_time, "w") as file:
                file.write(text)

        if args.graph is not None:
            draw_he(args.graph, *heat_values)
