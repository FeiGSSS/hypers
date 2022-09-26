python main.py --data_name cora     --data_path data/raw_data/cocitation/ --num_layers 3
python main.py --data_name citeseer --data_path data/raw_data/cocitation/ --num_layers 3
python main.py --data_name pubmed   --data_path data/raw_data/cocitation/ --num_layers 3 --convs 

python main.py --data_name cora     --data_path data/raw_data/coauthorship/ --num_layers 3
python main.py --data_name dblp     --data_path data/raw_data/coauthorship/ --num_layers 3 --convs

python main.py --data_name zoo        --data_path data/raw_data/             --num_layers 3
python main.py --data_name ModelNet40 --data_path data/raw_data/             --num_layers 3 --convs
python main.py --data_name NTU2012    --data_path data/raw_data/             --num_layers 2
# python main.py --data_name 20newsW100 --data_path data/raw_data/             --num_layers 3
# python main.py --data_name yelp --data_path data/raw_data/             --num_layers 3