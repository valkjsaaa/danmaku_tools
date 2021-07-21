#!/usr/bin/env python
#
# Script for keyword analysis of Bilibili livestream danmaku
#
# Code by @kumafans (Bilibili ID: OneTwoInfinity)
#
# Built upon danmaku_tools project of valkjsaaa
#
#
# Suggested usage:
#
#   python danmaku_keyword.py foo.xml --he_keyword_map foo.txt
#
# For debug or development:
#
#   python danmaku_keyword.py foo.xml --he_keyword_map foo.txt --he_keyword_graph foo.png
#
#
####

import argparse

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Droid Sans Fallback']
import re
from scipy.signal import find_peaks

import jieba
jieba.load_userdict('userdict.txt')
from collections import Counter
import math
import operator
from datetime import timedelta

from danmaku_tools.danmaku_tools import read_danmaku_file, get_value, get_time


parser = argparse.ArgumentParser(description='Process bilibili Danmaku')
parser.add_argument('danmaku', type=str, help='path to the danmaku file')
parser.add_argument('--he_keyword_graph', type=str, default=None, help='output graph path, leave empty if not needed')
parser.add_argument('--he_keyword_map', type=str, default=None, help='output high density timestamp, leave empty if not needed')


def preprocess_danmaku(danmaku):
    
    """
    Some ad-hoc processing for danmaku with repeated patterns
    to allow more consistent word splitting by jieba

    Note:
    Current rules are based on usage patterns in live.bilibili.com/261
    may require tuning for other livestreams

    Keyword arguments:
    danmaku -- danmaku string

    Returns:
    danmaku -- danmaku string after processing
    """
    danmaku = re.sub(r'^7777+$', '7777777', danmaku)
    danmaku = re.sub(r'^5555+$', '55555', danmaku)
    danmaku = re.sub(r'^88+$', '88', danmaku)
    danmaku = re.sub(r'^哈哈哈+$', '哈哈哈', danmaku)
    danmaku = re.sub(r'^？+$', '？', danmaku)
    danmaku = re.sub(r'^\?+$', '？', danmaku)
    danmaku = re.sub(r'^(?:☀️)+$', '☀️', danmaku)
    danmaku = re.sub(r'^(?:☀)+$', '☀️', danmaku)
   
    return danmaku


def load_danmaku_xml(xml_tree):

    """
    Extract the list of danmaku and timestamps from the recorded xml file

    Keyword arguments:
    xml_tree -- xml tree of the recorded xml file

    Returns:
    danmaku_list -- list of danmaku, each entry as (danmaku_string, timestamp)
    """
    danmaku_list = []

    for entry in xml_tree:
        if entry.tag == 'd':
            danmaku_list.append((preprocess_danmaku(entry.text), int(get_time(entry))))

    return danmaku_list


def gen_danmaku_slices(danmaku_list, interval=30):

    """
    Convert the danmaku list into slices with the prespecified interval

    Keyword arguments:
    danmaku_list -- list of danmaku, each entry as (danmaku_string, timestamp)
    interval -- length of each slice (unit: seconds)

    Returns:
    slices -- slices of danmaku, each entry as a list of danmaku_string within the interval
    """
    interval = int(interval)
    final_time = danmaku_list[-1][1]
    slices = [[] for i in range(final_time//int(interval) + 1)]

    for entry in danmaku_list:
        slices[entry[1]//interval].append(entry[0])

    return slices


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
    that occurs in the xml

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


def gen_heat_values(wordcount_slices, idf_list, window_size=3):

    """
    Generate the heat value for all the slices
    Heat value is computed as the sum of word counts in the slice weighted by IDF

    Keyword arguments:
    wordcount_slices -- list of Counters, each Counter holds the word counts within the slice
    idf_list -- a dictionary with each word as the key and the IDF as the value
    window_size -- the width of window used for moving average

    Returns:
    heat_values -- list of computed heat values for the slices
    """
    num_slices = len(wordcount_slices)
    heat_values = [0]*num_slices

    for slice_id in range(num_slices):
        low = max(slice_id - (window_size-1)//2, 0)
        high = min(slice_id + (window_size)//2 + 1, num_slices)

        window = sum(wordcount_slices[low:high], Counter())
        for word in window:
            if word == ' ':
                continue
            heat_values[slice_id] += (window[word] * idf_list[word]) / (high-low)

    return np.array(heat_values)


def find_he_peaks(heat_values):

    """
    Identify the peaks in the given trace of heat values

    Note:
    Currently "prominence" is used as the criterion

    Keyword arguments:
    heat_values -- list of computed heat values for the slices

    Returns:
    peaks -- list of slice ID that are considered as peaks
    """
    threshold = max(heat_values)/10 ### ad-hoc
    peaks, properties = find_peaks(heat_values, prominence = threshold)

    return peaks


def peaks_to_range(heat_values, peaks):

    """
    Identify the time range that corresponds to the identified peaks

    Note:
    An ad-hoc approach is taken, basically following the monotonic decrease
    of heat values before the peak (but discard the lowest one), and
    optionally take one extra slice after the peak

    Keyword arguments:
    heat_values -- list of computed heat values for the slices
    peaks -- list of slice ID that are considered as peaks

    Returns:
    he_list -- list of (start, end) indicating the ranges
    top_he_id -- the ID of high energy range that correspond to the highest peak
    is_he -- a 0-1 array indicating whether a slice is high energy or not
    """
    is_he = np.zeros(len(heat_values))
    he_list = []

    for peak in peaks:
        low = peak-1
        high = peak
        while (low-1 >= 0) and (heat_values[low-1] < heat_values[low]):
            low -= 1
        if heat_values[peak+1] > heat_values[peak-1]:
            high = peak + 1
        is_he[low+1:high+1] = 1
        he_list.append((low+1, high))
        if peak == np.argmax(heat_values):
            top_he_id = len(he_list) - 1

    return he_list, top_he_id, is_he


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
    he_window = sum(wordcount_slices[he_range[0]:he_range[1]+1], Counter())
    word_importance = {}
    for word in he_window:
        if word == ' ':
            continue
        word_importance[word] = he_window[word] * idf_list[word]

    return sorted(word_importance.items(), key=operator.itemgetter(1))[-n_keys:][::-1]


def range_to_timestamp(he, interval):

    """
    Convert the high energy range (in slice ID) into time
    """
    ann_timestamp = str(timedelta(seconds=int(he[0])*interval))    \
            + " - " + str(timedelta(seconds=int(he[1]+1)*interval))

    return ann_timestamp


def gen_he_keyword_list(he_list, wordcount_slices, idf_dict, interval):

    """
    Generate the top keywords for all high energy slices

    Keyword arguments:
    he_list -- list of (start, end) indicating the ranges
    wordcount_slices -- a dictionary with each word as the key and the IDF as the value
    idf_list -- a dictionary with each word as the key and the IDF as the value
    interval -- length of each slice (unit: seconds)

    Returns:
    he_keyword_list -- list of (timestamp, top keywords) for all high energy slices
    """
    he_keyword_list = []

    for he in he_list:
        he_keyword_list.append([range_to_timestamp(he, interval), [keyword[0] for keyword in find_keywords(wordcount_slices, idf_dict, he, n_keys=3)]])

    return he_keyword_list


def segment_text(text):

    """
    From danmaku_tools.danmaku_energy_map
    """
    TEXT_LIMIT = 900
    SEG_CHAR = '\n\n\n\n'

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


def gen_he_keyword_file(fname, he_keyword_list, top_he_id, interval):

    """
    Generate keyword file, similar to danmaku_tools.danmaku_energy_map
    """
    if len(he_keyword_list) == 0:
        text = "没有高能..."
    else:

        text = f"全场关键词最高能：\n {he_keyword_list[top_he_id][0]}\t{','.join(he_keyword_list[top_he_id][1])}\n\n其他关键词高能："

        for he_line in he_keyword_list:
            text += f"\n {he_line[0]}\t{'、'.join(he_line[1])}"
        text += "\n\n\n\n"
        text += "自动热词定位由@OneTwoInfinity 友情提供\n目前仍在测试阶段，有问题或建议欢迎回复本评论或私信作者"
        text = segment_text(text)

        with open(fname, "w") as file:
            file.write(text)
        print(text)

    return


def draw_he(he_graph, heat_values, keyword_list, peaks):

    """
    Generate heat values plot

    Note:
    Currently only suitable for debug or internal visualization, not for production
    """
    fig = plt.figure(figsize=(16, 10), dpi=100)

    plt.plot([i*interval/60 for i in range(len(heat_values))], heat_values)
    plt.scatter([i*interval/60 for i in range(len(heat_values))], heat_values, c=is_he)

    for he, peak in zip(keyword_list, peaks):
        ann_text = he[0] + '\n'
        ann_text += '\n'.join(he[1])
        plt.annotate(ann_text, (peak*interval/60, heat_values[peak]))

    plt.xlim(0, len(heat_values)*interval/60)

    plt.savefig(he_graph)
    plt.show()


if __name__ == '__main__':

    """
    Workflow:

    1) load recorded XML (+ preprocessing)
    2) compute IDF for keywords
    3) compute weighted heat values
    4) identify peaks and high energy ranges
    5) get the keywords in the ranges
    6) output txt or debug figure
    """

    print("Starting to identify top keywords...")

    interval = 30

    # 1)
    args = parser.parse_args()
    xml_list = read_danmaku_file(args.danmaku)
    danmaku_list = load_danmaku_xml(xml_list)

    # 2)
    danmaku_slices = gen_danmaku_slices(danmaku_list, interval)
    wordcount_slices = gen_slice_wordcount(danmaku_slices)
    idf_dict = gen_idf_dict(wordcount_slices)

    # 3)
    heat_values = gen_heat_values(wordcount_slices, idf_dict, window_size=3)

    # 4)
    peaks = find_he_peaks(heat_values)
    he_list, top_he_id, is_he = peaks_to_range(heat_values, peaks)

    # 5)
    he_keyword_list = gen_he_keyword_list(he_list, wordcount_slices, idf_dict, interval)

    # 6)
    if args.he_keyword_map:
        gen_he_keyword_file(args.he_keyword_map, he_keyword_list, top_he_id, interval)
    if args.he_keyword_graph:
        draw_he(args.he_keyword_graph, heat_values, he_keyword_list, peaks)


