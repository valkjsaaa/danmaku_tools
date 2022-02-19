import argparse
import json
import xml.etree.ElementTree as ET

parser = argparse.ArgumentParser(description='Clean BiliLiveReocrder XML')
parser.add_argument('xml_files', type=str, help='path to the danmaku file')
parser.add_argument('--keep-lottery', dest='remove_lottery', action='store_false')
parser.add_argument('--remove-lottery', dest='remove_lottery', action='store_true')
parser.set_defaults(remove_lottery=True)
parser.add_argument('--output', type=str, default=None, help='output path for the output XML', required=True)


def process_root(orig_root, remove_lottery):
    new_root = ET.Element('i')
    for child in orig_root:
        if (child.tag in ['d']) and remove_lottery:
            danmaku_raw = json.loads(child.attrib['raw'])
            if type(danmaku_raw) is not list:  # mostly broadcasting messages
                new_root.append(child)
                break
            sender_id = danmaku_raw[0][5]
            if sender_id != 0:
                new_root.append(child)
        else:
            new_root.append(child)
    return new_root


if __name__ == '__main__':
    args = parser.parse_args()

    tree = ET.parse(args.xml_files)
    root = tree.getroot()

    new_root = process_root(root, args.remove_lottery)

    ET.ElementTree(new_root).write(args.output, encoding='UTF-8', xml_declaration=True)
