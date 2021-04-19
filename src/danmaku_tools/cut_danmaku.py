#!/usr/bin/env python

import argparse
import xml.etree.ElementTree as ET

from dateutil.parser import parse

parser = argparse.ArgumentParser(description='Merge BiliLiveReocrder XML')
parser.add_argument('xml_files', type=str, help='path to the danmaku file')
parser.add_argument('--start_time', type=float, help='use video length as the duration of the file', default=0)
parser.add_argument('--end_time', type=float, help='use video length as the duration of the file', default=float('inf'))
parser.add_argument('--output', type=str, default=None, help='output path for the output XML', required=True)


def get_root_time(root_xml):
    record_info = root_xml.findall('BililiveRecorderRecordInfo')[0]
    record_start_time_str = record_info.attrib['start_time']
    record_start_time = parse(record_start_time_str)
    return record_start_time


def process_root(orig_root, start_time, end_time):
    new_root = ET.Element('i')
    for child in orig_root:
        if child.tag in ['sc', 'gift', 'guard']:
            orig_time = float(child.attrib['ts'])
            if start_time <= orig_time <= end_time:
                new_time = orig_time - start_time
                new_time_str = str(new_time)
                child.set('ts', new_time_str)
                new_root.append(child)
        elif child.tag in ['d']:
            orig_parameters_str = child.attrib['p'].split(',')
            orig_time = float(orig_parameters_str[0])
            if start_time <= orig_time <= end_time:
                new_time = orig_time - start_time
                new_parameters_str = [str(new_time)] + orig_parameters_str[1:]
                child.set('p', ','.join(new_parameters_str))
                new_root.append(child)
        else:
            new_root.append(child)
    return new_root


if __name__ == '__main__':
    args = parser.parse_args()

    tree = ET.parse(args.xml_files)
    root = tree.getroot()
    new_root_offset = 0
    root_time = get_root_time(root)
    all_flv = ""

    new_root = process_root(root, args.start_time, args.end_time)

    ET.ElementTree(new_root).write(args.output, encoding='UTF-8', xml_declaration=True)
