import sys
segments_file = sys.argv[1]
wav_scp_file = sys.argv[2]
wavid2path = {}
with open(wav_scp_file,'r') as f:
    for line in f.readlines():
        line = line.strip().split()
        wavid2path[line[0]] = line[1]

with open(segments_file,'r') as f:
    for line in f.readlines():
        line = line.strip().split()
        print('{} {},{},{}'.format(line[0],wavid2path[line[1]],line[2],line[3]))
