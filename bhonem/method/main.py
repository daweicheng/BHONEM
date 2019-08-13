import sys
#print("\n in main.py\n",sys.path)
sys.path.append(sys.path[0]+"/libnrl")
sys.path.append(sys.path[0]+"/libnrl/gcn")
import numpy as np
import tensorflow as tf
import random
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from sklearn.linear_model import LogisticRegression
from libnrl.graph import *
from libnrl import node2vec
from libnrl.classify import Classifier, read_node_label, read_node_label_deleted, load_embeddings
from libnrl import line
#from libnrl import tadw
#from libnrl.gcn import gcnAPI
#from libnrl.grarep import GraRep
import time

def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')
   
    parser.add_argument('--delete-node',default='',
                        help='Delete the guarantee nodes')
    parser.add_argument('--deleted', default=0,
                        help='value=1,Delete the guarantee nodes,')
    
    parser.add_argument('--input', required=True,
                        help='Input graph file')
    parser.add_argument('--output', required=True,
                        help='Output representation file')
    
    parser.add_argument('--directed', action='store_true',
                        help='Treat graph as directed.')
    
    parser.add_argument('--representation-size', default=128, type=int,
                        help='Number of latent dimensions to learn for each node.')
    
    parser.add_argument('--epochs', default=5, type=int,
                        help='The training epochs of LINE and GCN')
    
    parser.add_argument('--method', required=True, choices=['node2vec', 'deepWalk', 'line', 'gcn', 'grarep', 'tadw'],
                        help='The learning method')
    parser.add_argument('--label-file', default='',
                        help='The file of node label')
    
    parser.add_argument('--graph-format', default='adjlist', choices=['adjlist', 'edgelist'],
                        help='Input graph format')
    
    parser.add_argument('--weighted', action='store_true',
                        help='Treat graph as weighted')
    parser.add_argument('--clf-ratio', default=0.5, type=float,
                        help='The ratio of training data in the classification')
    
    parser.add_argument('--no-auto-stop', action='store_true',
                        help='no early stop when training LINE')
    parser.add_argument('--order', default=3, type=int,
                        help='Choose the order of LINE, 1 means first order, 2 means second order, 3 means first order + second order')
    
    args = parser.parse_args()
    return args


def main(args):
    t1 = time.time()
    
    ### get delete nodes which are guarantee nodes , after embedde all the node in the net, we have to delete the embeddings of delete-nodes
    if args.deleted:
        delete_node = []        
        filename = args.delete_node 
        fin = open(filename,'r')        
        while 1:            
            l = fin.readline()
            if l == '':
                break
            vec = l.strip().split(' ')
            temp = [str(x) for x in vec[0:]]
            delete_node.append(temp[0])        
        fin.close()
    
    g = Graph()
    print("Reading...")
    if args.graph_format == 'adjlist':
        g.read_adjlist(filename=args.input)
    elif args.graph_format == 'edgelist': # our data are all in the type of edge_list
        g.read_edgelist(filename=args.input, weighted=args.weighted, directed=args.directed)
    
    if args.method == 'line':
        if args.label_file and not args.no_auto_stop:
            model = line.LINE(g, epoch = args.epochs, rep_size=args.representation_size, order=args.order, 
                label_file=args.label_file, clf_ratio=args.clf_ratio)
        else:
            model = line.LINE(g, epoch = args.epochs, rep_size=args.representation_size, order=3)
    
    t2 = time.time()
    print(t2-t1)
    
    print("Saving embeddings...")
    if args.deleted:
        model.save_embeddings_deleted(args.output,delete_node)
    else:
        model.save_embeddings(args.output)

    if args.deleted:
        X_model, Y_model = read_node_label_deleted(args.label_file,delete_node)
    else:
        X_model, Y_model = read_node_label(args.label_file)
    
    vectors = load_embeddings(args.output)  
    clf = Classifier(vectors=vectors, clf=LogisticRegression())
    clf.split_train_evaluate(X_model, Y_model, args.clf_ratio)
    

if __name__ == "__main__":
    random.seed(32)
    np.random.seed(32)
    config = tf.ConfigProto() 
    config.gpu_options.per_process_gpu_memory_fraction = 0.8 # 占用GPU40%的显存 
    session = tf.Session(config=config)
    main(parse_args())

