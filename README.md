Bilibili Live Danmaku Tools 哔哩哔哩直播弹幕处理工具
===========================

本工具可用于处理 B站录播姬 产生的弹幕 XML。具体功能有：
1. 分切 弹幕 XML 
2. 合并 弹幕 XML 
3. 分析 弹幕 XML 中的弹幕数量以及礼物价格
4. 分析 弹幕 XML 中的高能点（主要用于生成录播）

### 安装

```bash
pip3 install danmaku_tools
```

### 典型使用例子

#### 合并

根据 flv 文件的长度合并 XML
```bash
python3 -m danmaku_tools.merge_danmaku video_1.xml video_2.xml video_3.xml --video_time ".flv" --output video_combined.xml
```

经常和类似这样的视频合并命令同时使用
```bash
echo "file video_1.flv\n file video_2.flv" > video.input.txt
ffmpeg -f concat -safe 0 -i video_input.txt video_combined.flv
```

根据 XML 开始时间合并 XML
```bash
python3 -m danmaku_tools.merge_danmaku video_1.xml video_2.xml video_3.xml --output video_combined.xml
```

#### 剪切

从 123.45 秒开始剪切 XML
```bash
python3 -m danmaku_tools.cut_danmaku --start_time 123.45 video_input.xml --output video_output.xml
```

从 123.45 秒到 567.89 开始剪切 XML
```bash
python3 -m danmaku_tools.cut_danmaku --start_time 123.45 --end_time 567.89 video_input.xml --output video_output.xml
```

经常和类似这样的视频剪切命令同时使用
```bash
ffmpeg -ss 123.45 -to 567.89 -i video_input.flv video_output.flv
```

#### 分析

```bash
danmaku_energy_map.py video.xml `# 输入 XML 文件` \
  --graph video.he.png `# 高能进度条 png` \
  --he_map he_list.txt `# 高能列表` \
  --sc_list sc_list.txt `# 醒目留言列表` \
  --sc_srt sc.srt `# 醒目留言字幕` \
  --he_time he_time.txt `# 最高能时间点`
```





