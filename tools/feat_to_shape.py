import sys
import kaldiio
feats_scp=sys.argv[1]
shape_scp = sys.argv[2]
f_out = open(shape_scp,'w')
with open(feats_scp,'r') as f:
    for line in f.readlines():
        line = line.strip().split()
        uttid = line[0]
        feat_path = line[1]
        mat = kaldiio.load_mat(feat_path)
        length=mat.shape[0]
        dim=mat.shape[1]
        f_out.write(uttid+' '+str(length)+','+str(dim)+'\n')
f_out.close()
