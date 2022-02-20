#!/usr/bin/env python

####
#
#   Update on 20220219 by @kumafans (Bilibili ID: OneTwoInfinity)
#       - Use IDF-weighted keyword value to rate danmaku
#       - Add markers for Super Chat
#       - Eliminated the empirical 45 second offset in most decisions
#
####

import argparse
import json
import traceback
import random
from datetime import timedelta

import srt
import xml.etree.ElementTree as ET
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.filters import convolve
from scipy.stats import halfnorm

from danmaku_tools.danmaku_tools import read_danmaku_file, get_value, get_time

import jieba
from collections import Counter
import math
import re
import operator

from textrank4zh import TextRank4Sentence
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Process bilibili Danmaku')
parser.add_argument('danmaku', type=str, help='path to the danmaku file')
parser.add_argument('--graph', type=str, default=None, help='output graph path, leave empty if not needed')
parser.add_argument('--he_map', type=str, default=None, help='output high density timestamp, leave empty if not needed')
parser.add_argument('--sc_list', type=str, default=None, help='output super chats, leave empty if not needed')
parser.add_argument('--sc_srt', type=str, default=None, help='output super chats srt, leave empty if not needed')
parser.add_argument('--he_time', type=str, default=None, help='output highest density timestamp, leave empty if not '
                                                              'needed')
parser.add_argument('--he_range', type=str, default=None, help='output he_range, leave empty if not needed')
parser.add_argument('--user_xml', type=str, default=None, help='output danmaku xml with username, leave empty if not '
                                                               'needed')

parser.add_argument('--user_dict', type=str, default=None, help='[for he label generation] user-defined keywords, '
                                                                'leave empty if not needed')
parser.add_argument('--regex_rules', type=str, default=None, help='[for he label generation] table for regex rules, '
                                                                  'leave empty if not needed')



def preprocess_danmaku(danmaku):

    """
    Some ad-hoc processing for danmaku with repeated patterns
    to allow more consistent word splitting by jieba,
    substitution rules are specified using --regex_rules

    Keyword arguments:
    danmaku -- danmaku string

    Returns:
    danmaku -- danmaku string after processing
    """
    if args.regex_rules is not None:
        for line in open(args.regex_rules, 'r'):
            line = line.strip().split()
            assert(len(line) == 2)
            a = danmaku
            danmaku = re.sub(r"{}".format(line[0]), line[1], danmaku)

    return danmaku


def gen_slice_wordcount(danmaku_slices):

    """
    Count the occurence of words in each danmaku slice

    Keyword arguments:
    danmaku_slices -- slices of danmaku, each entry as a list of danmaku_string within the interval

    Returns:
    list of Counters, each Counter holds the word counts within the slice
    """
    return [Counter(jieba.cut(" ".join(slice))) for slice in danmaku_slices]


def gen_idf_dict(wordcount_slices):

    """
    Generate the inverse document frequency (IDF) for all words
    that occurs in the xml (danmaku file)

    Keyword arguments:
    wordcount_slices -- list of Counters, each Counter holds the word counts within the slice

    Returns:
    idf_list -- a dictionary with each word as the key and the IDF as the value
    """
    all_words = list(sum(wordcount_slices, Counter()))
    idf_list = {}

    for word in all_words:
        idf_list[word] = math.log(len(wordcount_slices)
                  /  (1+ sum(1 for slice in wordcount_slices if word in slice)))

    return idf_list


def gen_danmaku_slices(all_children, interval=1):

    """
    Convert the danmaku list into slices with the prespecified interval

    Keyword arguments:
    danmaku_list -- list of danmaku, each entry as (danmaku_string, timestamp)
    interval -- length of each slice (unit: seconds)

    Returns:
    slices -- slices of danmaku, each entry as a list of danmaku_string within the interval
    """
    interval = int(interval)
    final_time = get_time(all_children[-1])
    slices = [[] for i in range(int(final_time)//int(interval) + 1)]

    for child in all_children:
        if child.tag == 'd':
            if json.loads(child.attrib['raw'])[0][5] != 0: ### not lucky draw danmaku
                slices[int(get_time(child))//interval].append(preprocess_danmaku(child.text))

    return slices


def get_danmaku_value(cur_danmaku, idf_list):

    """
    Get the IDF-weighted value of a given danmaku

    Keyword arguments:
    cur_danmaku -- XML element tree child
    idf_list -- precomputed dictionary containing IDF for all keywords

    Returns:
    danmaku_value -- value of the input danmaku, calculated by summing up the IDF weights of all keywords
    """
    danmaku_value = 0
    if cur_danmaku.tag == 'd':
        if json.loads(cur_danmaku.attrib['raw'])[0][5] != 0: ### not lucky draw danmaku
            for word in jieba.cut(preprocess_danmaku(cur_danmaku.text)):
                danmaku_value += idf_list[word]

    return danmaku_value


def half_gaussian_filter(value, sigma):
    space = np.linspace(-4, 4, sigma * 8)
    neg_space = np.linspace(4 * 10, -4 * 10, sigma * 8)
    kernel = (halfnorm.pdf(space) + halfnorm.pdf(neg_space)) / sigma
    offset_value = np.concatenate((value, np.zeros(45)))
    return convolve(offset_value, kernel)[45:]


def get_heat_time(all_children, idf_list):
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
            cur_heat += get_danmaku_value(cur_danmaku, idf_list)
            cur_entry += 1

        while len(danmaku_queue) != 0 and get_time(danmaku_queue[0]) < start:
            prev_danmaku = danmaku_queue.popleft()
            cur_heat -= get_danmaku_value(prev_danmaku, idf_list)

        heat_time[0] += [center]
        heat_time[1] += [cur_heat]
        center += 1

    heat_value = heat_time[1]
    heat_value_gaussian = half_gaussian_filter(heat_value, sigma=50)
    heat_value_gaussian2 = half_gaussian_filter(heat_value, sigma=1000) * 1.2

    he_points = [[], []]
    cur_highest = -1
    highest_idx = -1
    he_start = -1
    he_range = []

    for i in range(len(heat_value_gaussian)):
        if highest_idx != -1:
            assert he_start != -1
            if heat_value_gaussian[i] < heat_value_gaussian2[i]:
                he_points[0] += [highest_idx]
                he_points[1] += [cur_highest]
                he_range += [(he_start, i)]
                highest_idx = -1
                he_start = -1
            else:
                if heat_value_gaussian[i] / np.sqrt(heat_value_gaussian2[i]) > cur_highest:
                    cur_highest = heat_value_gaussian[i] / np.sqrt(heat_value_gaussian2[i])  ### changed to match with the graph
                    highest_idx = i
        else:
            assert he_start == -1
            if heat_value_gaussian[i] > heat_value_gaussian2[i]:
                cur_highest = heat_value_gaussian[i] / np.sqrt(heat_value_gaussian2[i])  ### changed to match with the graph
                highest_idx = i
                he_start = i

    # Usually the HE point at the end of a live stream is just to say goodbye
    # if highest_idx != -1:
    #     he_points[0] += [highest_idx]
    #     he_points[1] += [cur_highest]

    return heat_time, heat_value_gaussian / np.sqrt(heat_value_gaussian2), np.sqrt(
        heat_value_gaussian2), he_points, he_range


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


def draw_he(he_graph, heat_time, heat_value_gaussian, heat_value_gaussian2, he_points, he_range, current_time=-1, sc_tuple=None):
    # sc_tuple = (time, price, message, user, duration)
    fig = plt.figure(figsize=(16, 1), frameon=False, dpi=60)
    ax = fig.add_axes((0, 0, 1, 1))
    draw_he_area(ax, current_time, heat_time, heat_value_gaussian, heat_value_gaussian2)
    # draw_he_annotate_line(ax, current_time, heat_time, he_points)
    if sc_tuple is not None and (len(sc_tuple) != 0):
        height = min(heat_value_gaussian) + 0.1*(max(heat_value_gaussian)-min(heat_value_gaussian))
        for sc in sorted(sc_tuple, key=lambda x: int(x[1])):
            # Order the SCs by price before plotting to prevent the expensive SCs being covered by cheaper ones
            sc_price = int(sc[1])
            if sc_price < 50:
                sc_color = (42,96,178)
            elif sc_price < 100:
                sc_color = (66, 125, 158)
            elif sc_price < 500:
                sc_color = (226, 181, 43)
            elif sc_price < 1000:
                sc_color = (224, 148, 67)
            elif sc_price < 2000:
                sc_color = (229, 77, 77)
            else:
                sc_color = (171, 26, 50)
            plt.scatter(sc[0], height, s=75, c=[[rgb/255.0 for rgb in sc_color]])
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


def get_danmaku_from_range(all_children, he_range):
    start, end = he_range
#    start += 45
#    end += 45
    return [item.text for item in all_children if item.tag == 'd' and start <= get_time(item) <= end]


def find_keywords(wordcount_slices, idf_list, he_range, n_keys=3):

    """
    Identify the top keywords for a given high energy slice

    Keyword arguments:
    wordcount_slices -- list of computed heat values for all the slices
    idf_list -- a dictionary with each word as the key and the IDF as the value
    he_range -- (start, end) indicating one high energy range
    n_keys -- number of keywords to output

    Returns:
    a list of top keywords in the specified high energy range
    """
    he_window = sum(wordcount_slices[he_range[0]:he_range[1]+1], Counter())   # Empirical offset (45s) removed
    word_importance = {}
    for word in he_window:
        if word == ' ':
            continue
        word_importance[word] = he_window[word] * idf_list[word]

    return sorted(word_importance.items(), key=operator.itemgetter(1))[-n_keys:][::-1]



if __name__ == '__main__':
    args = parser.parse_args()
    xml_list = read_danmaku_file(args.danmaku)

    if args.sc_list is not None or args.sc_srt is not None:
        sc_chats = [element for element in xml_list if element.tag == 'sc']

        sc_tuple = []
        for sc_chat_element in sc_chats:
            try:
                price = sc_chat_element.attrib['price']
                raw_message = json.loads(sc_chat_element.attrib['raw'])
                message = raw_message["message"].replace('\n', '\t')
                user = raw_message["user_info"]['uname']
                time = float(sc_chat_element.attrib['ts'])
                duration = raw_message['time']
                sc_tuple += [(time, price, message, user, duration)]
            except:
                print(f"superchat processing error {sc_chat_element}")

        if args.sc_list is not None:
            if len(sc_tuple) != 0:
                sc_text = "醒目留言列表："
                for time, price, message, user, _ in sc_tuple:
                    sc_text += f"\n {convert_time(int(time))} ¥{price} {user}: {message}"
                sc_text += "\n"
                sc_text = segment_text(sc_text)
            else:
                sc_text = "没有醒目留言..."
            with open(args.sc_list, "w") as file:
                file.write(sc_text)
        if args.sc_srt is not None:
            active_sc = []
            subtitles = []
            cur_time = 0


            def display_sc(start, end, sc_list):
                display_sorted_sc = sorted(sc_list, key=lambda x: (-float(x[0]), -int(x[2])))
                content = "\n".join([sc[3] for sc in display_sorted_sc])
                LIMIT = 100
                if len(content) >= LIMIT:
                    content = content[:LIMIT - 2] + "…"
                return srt.Subtitle(
                    index=0,
                    start=timedelta(seconds=start),
                    end=timedelta(seconds=end),
                    content=content
                )


            def flush_sc(start_time: float, end_time: float):
                current_sc = sorted(active_sc, key=lambda x: x[1])
                subtitle_list = []
                while True:
                    if len(current_sc) == 0:
                        break
                    if current_sc[0][1] < end_time:
                        if current_sc[0][1] - start_time > 1:
                            subtitle_list += [display_sc(start_time, current_sc[0][1], current_sc)]
                            start_time = current_sc[0][1]
                    else:
                        break
                    current_sc.pop(0)
                if end_time - start_time > 1:
                    subtitle_list += [display_sc(start_time, end_time, current_sc)]
                    start_time = end_time
                return current_sc, subtitle_list, start_time


            for time, price, message, user, duration in sc_tuple:
                start = time
                end = time + duration * 0.6
                content = f"¥{price} {user}: {message}".replace("绑架", "**")
                new_sc, new_subtitles, cur_time = flush_sc(start_time=cur_time,
                                                           end_time=start)  # Flush all the previous SCs
                active_sc = new_sc + [(start, end, price, content)]
                subtitles += new_subtitles
            if len(active_sc):
                end_time = max([sc[1] for sc in active_sc])
                _, new_subtitles, _ = flush_sc(start_time=cur_time, end_time=end_time)
                subtitles += new_subtitles
            with open(args.sc_srt, "w") as file:
                file.write(srt.compose(subtitles))

    if args.he_map is not None or args.graph is not None or args.he_time is not None or args.he_range:
        if args.user_dict is not None:
            jieba.load_userdict(args.user_dict)
            print(f"User-defined dictionary '{args.user_dict}' loaded")
        slices = gen_danmaku_slices(xml_list, 1)
        wordcount_slices = gen_slice_wordcount(slices)
        idf_list = gen_idf_dict(wordcount_slices)

        heat_values = get_heat_time(xml_list, idf_list)

        if args.he_range is not None:
            with open(args.he_range, "w") as file:
                json.dump(heat_values[4], file)

        if args.he_map is not None:
            he_pairs = heat_values[3]
            all_timestamps = heat_values[0][0]

            heat_comments = []
            xml_list_iter = iter(xml_list)
            tr4s = TextRank4Sentence()
            for start, end in tqdm(heat_values[4]):
                comment_list = []
                while True:
                    try:
                        element = next(xml_list_iter)
                    except StopIteration:
                        break
                    if get_time(element) <= start + 0:    # Empirical offset (45s) removed
                        continue
                    if get_time(element) > end + 0:    # Empirical offset (45s) removed
                        break
                    if element.tag == 'd':
                        text = element.text
                        if text is not None and not text.replace(" ", "").replace("哈", "") == "":
                            comment_list += [text]
                print(len(comment_list))
                if len(comment_list) > 1000:
                    comment_list = random.sample(comment_list, 1000)
                tr4s.analyze("\n".join(comment_list), lower=True, source='no_filter')
                key_sentences = tr4s.get_key_sentences(num=1, sentence_min_len=1)
                if len(key_sentences) > 0:
                    top_sentence = key_sentences[0]['sentence']
                else:
                    top_sentence = ""
                heat_comments += [top_sentence]

            if len(he_pairs[0]) == 0:
                text = "没有高能..."
            else:
                # noinspection PyTypeChecker
                highest_id = np.argmax(he_pairs[1])
                highest_time_id = he_pairs[0][highest_id]
                highest_time = all_timestamps[highest_time_id]
                highest_sentence = heat_comments[highest_id]

                other_he_time_list = [all_timestamps[time_id] for time_id in he_pairs[0]]

                text = f"全场最高能：{convert_time(highest_time)}\t{highest_sentence}\n\n其他高能："

                for i, (start_he_time, end_he_time) in enumerate(heat_values[4]):
                    text += f"\n {convert_time(start_he_time)} - {convert_time(end_he_time)}\t{heat_comments[i]}\t"
                    text += "("
                    text += ",".join([kw for kw, value in \
                            find_keywords(wordcount_slices, idf_list, (start_he_time, end_he_time), n_keys=3)])
                    text += ")"

            text += "\n"
            text = segment_text(text)
            with open(args.he_map, "w") as file:
                file.write(text)

        if args.he_time is not None:
            he_pairs = heat_values[3]
            all_timestamps = heat_values[0][0]
            if len(he_pairs[0]) == 0:
                text = "0"
            else:
                # noinspection PyTypeChecker
                highest_time_id = he_pairs[0][np.argmax(he_pairs[1])]
                highest_time = all_timestamps[highest_time_id]
                text = str(highest_time)
            with open(args.he_time, "w") as file:
                file.write(text)

        if args.graph is not None:
            if args.sc_list is not None or args.sc_srt is not None:
                draw_he(args.graph, *heat_values, sc_tuple=sc_tuple)
            else:
                draw_he(args.graph, *heat_values)

    if args.user_xml is not None:
        tree = ET.parse(args.danmaku)
        user_cache = {}


        def get_user_follower(user_id):
            if user_id in user_cache:
                return user_cache[user_id]
            else:
                import bilibili_api
                user_follower = bilibili_api.user.get_relation_info(user_id)['follower']
                user_cache[user_id] = user_follower
                return user_follower


        for child in tqdm(tree.getroot()):
            try:
                if child.tag == 'd':
                    user_name = child.attrib['user']
                    raw_data = json.loads(child.attrib['raw'])
                    user_id = raw_data[2][0]
                    # follower = get_user_follower(user_id)
                    follower = 0
                    user_level = raw_data[3][0] if len(raw_data[3]) > 0 else 0
                    user_boat = raw_data[7]
                    display_username = follower >= 1000 or user_level > 25 or user_boat >= 2
                    if display_username:
                        print(user_name)
                        child.text = f"@{user_name}:" + child.text
            except Exception as e:
                print(e)
                print(traceback.format_exc())
        tree.write(args.user_xml, encoding='UTF-8', xml_declaration=True)
