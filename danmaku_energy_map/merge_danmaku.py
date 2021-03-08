import argparse
import xml.etree.ElementTree as ET
from dateutil.parser import parse

parser = argparse.ArgumentParser(description='Merge BiliLiveReocrder XML')
parser.add_argument('xml_files', type=str, nargs='+', help='path to the danmaku file')
parser.add_argument('--output', type=str, default=None, help='output path for the output XML', required=True)

def get_root_time(root_xml):
    record_info = root_xml.findall('BililiveRecorderRecordInfo')[0]
    record_start_time_str = record_info.attrib['start_time']
    record_start_time = parse(record_start_time_str)
    return record_start_time


def add_root(orig_root, new_root, new_offset):
    for child in new_root:
        if child.tag in ['sc', 'gift', 'guard']:
            orig_time = float(child.attrib['ts'])
            new_time = orig_time + new_offset
            new_time_str = str(new_time)
            child.set('ts', new_time_str)
            orig_root.append(child)
        if child.tag in ['d']:
            orig_parameters_str = child.attrib['p'].split(',')
            orig_time = float(orig_parameters_str[0])
            new_time = orig_time + new_offset
            new_parameters_str = [str(new_time)] + orig_parameters_str[1:]
            child.set('p', ','.join(new_parameters_str))
            orig_root.append(child)


if __name__ == '__main__':
    args = parser.parse_args()

    if len(args.xml_files) == 0:
        print("At least one XML files have to be passed as input.")

    tree = ET.parse(args.xml_files[0])
    root = tree.getroot()
    root_time = get_root_time(root)

    for i in range(len(args.xml_files) - 1):
        new_root = ET.parse(args.xml_files[i + 1]).getroot()
        new_root_offset = (get_root_time(new_root) - root_time).total_seconds()
        add_root(root, new_root, new_root_offset)

    tree.write(args.output, encoding='UTF-8', xml_declaration=True)
