python main.py --data_name cora     --data_path data/raw_data/cocitation/ --dim 256 --heads 4 --convs gnn --device 3
python main.py --data_name citeseer --data_path data/raw_data/cocitation/ --dim 512 --heads 8 --convs gnn --device 3
python main.py --data_name pubmed   --data_path data/raw_data/cocitation/ --dim 256 --heads 8 --convs gnn --device 3

python main.py --data_name cora     --data_path data/raw_data/coauthorship/  --dim 128 --heads 8 --convs gnn --device 3
python main.py --data_name dblp     --data_path data/raw_data/coauthorship/  --dim 512 --heads 8 --convs gnn --device 3

python main.py --data_name zoo        --data_path data/raw_data/             --dim 64  --heads 1 --lr 0.01 --wd 1e-5 --convs gnn --device 3
python main.py --data_name ModelNet40 --data_path data/raw_data/             --dim 512 --heads 8 --convs gnn --device 3
python main.py --data_name NTU2012    --data_path data/raw_data/             --dim 256 --heads 1 --convs gnn --device 3
# python main.py --data_name 20newsW100 --data_path data/raw_data/     --convs        --dim 256
# python main.py --data_name yelp --data_path data/raw_data/             