import joblib
import numpy as np
import torch, dgl
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from .utils import *
import pandas as pd
from scipy.stats import spearmanr
class DiModel():
    def __init__(self, k, ss, TopN=None):

        print('Kmeans get Di sim of {df}')

        m_gs = ss

        self.scaler = StandardScaler()
        m_gs = self.scaler.fit_transform(m_gs)

        max_k = k
        k_range = range(1, max_k + 1)
        wss = []
        for k in k_range:
            kmeans = KMeans(n_clusters=k, init='k-means++', n_init=100, random_state=42)
            kmeans.fit(m_gs)
            wss.append(kmeans.inertia_)

        diff1 = np.diff(wss)
        diff2 = np.diff(diff1)
        elbow = np.argmax(diff2) + 1
        k = k_range[elbow]
        if TopN is None:
            TopN = 1
        self.kmeans_model = KMeans(n_clusters=k, init='k-means++', n_init=100)
        m_gc = self.kmeans_model.fit_transform(m_gs)
        m_gc_copy = np.copy(m_gc)
        min_two_idx = np.argsort(m_gc_copy, axis=1)[:, :TopN]
        m_gc_copy[:, :] = 0
        for i, idx in enumerate(min_two_idx):
            m_gc_copy[i, idx] = 1
        self.k = k
        self.TopN = TopN
        assert self.TopN <= k
        self.DiC = m_gc_copy

    def get_DiC(self):
        return self.DiC

    def transform(self, ss):
        if len(ss.shape) == 1:
            ss = ss.reshape((1,-1))
        m_gc = self.kmeans_model.transform(ss)
        m_gc_copy = np.copy(m_gc)
        min_two_idx = np.argsort(m_gc_copy, axis=1)[:, :self.TopN]
        m_gc_copy[:, :] = 0
        for i, idx in enumerate(min_two_idx):
            m_gc_copy[i, idx] = 1
        return m_gc_copy

class Config:

    def __init__(self, data_name='data', datapath=None, save_file=None,
                 save_model_file=None):
        self.data_name=data_name
        self.datapath = datapath
        self.save_file = save_file
        self.kfold = 5
        self.repeat = 10
        self.self_encode_len = 256
        self.globel_random = 42
        self.search_args = {'arg_name': [], 'arg_value': []}
        self.epochs = 1000
        self.print_epoch = 10

        self.lr = 0.001
        self.weight_decay = 1e-6

        self.patience = 30
        self.feature_perception = True
        self.test_size = 0.0
        self.val_test_size = 0.05
        self.save_model_file = save_model_file
        self.early_stop = True

        self.edge_emb_layer = 2
        self.gnn_layer = 2
        self.n_hidden = 32
        self.ncf_hidden = 128

        self.kmeans_max_K = 10
        self.dropout = 0.2

        self.neg_beta = 1
        self.noise_level = 0

        self.Use_GNN_Embedding = True
        self.Dynamic_train = False

        self.mask_sp = None
        self.new_sp = None

        self.save_ASS_Embedding = False
        self.edge_number = None

        self.kmeans_TopN = 1
        self.TopK_ss = 5
        set_seed(self.globel_random)

    def get_data_params(self, biodata):
        self.m_numb = biodata.m_numb
        self.d_numb = biodata.d_numb
        self.cm_numb = biodata.cm_numb
        self.cd_numb = biodata.cd_numb
        self.num_nodes = biodata.num_nodes
        self.num_rels = biodata.num_rels
        self.device = biodata.device

    def set_new_sp(self, new_sp):

        assert new_sp in ['RNA', 'disease', 'RNA-disease']
        self.new_sp = new_sp

    def reset_data_name(self, data_name):
        self.data_name = data_name

    def copy(self):
        return copy.deepcopy(self)

class BioData():
    def __init__(self, md_file=None, mss_list=None, dss_list=None,  m_encoding_list=[None,],
                 d_encoding_list=[None, ], x_encoding=False,  device='cpu', data_name='data',
                 config=Config(), sep=','):

            config.reset_data_name(data_name)
            self.config = config
            if md_file is None:
                self.ASS_Embedding = {

                }
            else:
                self.ASS_Embedding = {
                }

                self.num_rels = 0
                self.device = device
                self.metapath_name = {}
                self.TopK_ss = self.config.TopK_ss
                self.kmeans_max_K = self.config.kmeans_max_K
                self.kmeans_TopN = self.config.kmeans_TopN

                try:
                    self.adj_matrix = pd.read_csv(md_file, header=None, index_col=None, sep=sep).fillna(0).values
                except:
                    self.adj_matrix = pd.DataFrame(md_file).fillna(0).values
                mss_matrix_list = []
                dss_matrix_list = []
                if mss_list is None:
                    mss_matrix_list.append(randomMinMax(self.adj_matrix.shape[0]))
                else:
                    try:
                        for mss_f in mss_list:
                            mss_matrix_list.append(pd.read_csv(mss_f, header=None, index_col=None, sep=sep).fillna(0).values[np.newaxis, ...])
                    except:
                        for mss_f in mss_list:
                            mss_matrix_list.append(pd.DataFrame(mss_f).fillna(0).values[np.newaxis, ...])

                if dss_list is None:
                    mss_matrix_list.append(randomMinMax(self.adj_matrix.shape[1]))
                else:
                    try:
                        for dss_f in dss_list:
                            dss_matrix_list.append(pd.read_csv(dss_f, header=None, index_col=None, sep=sep).fillna(0).values[np.newaxis, ...])
                    except:
                        for dss_f in dss_list:
                            dss_matrix_list.append(pd.DataFrame(dss_f).fillna(0).values[np.newaxis, ...])

                self.mss_matrix_list = [ss.mean(0) for ss in mss_matrix_list]
                self.dss_matrix_list = [ss.mean(0) for ss in dss_matrix_list]
                self.mss = np.concatenate(mss_matrix_list, axis=0).mean(0) if len(mss_matrix_list) >= 2 else\
                                                                            mss_matrix_list[0][0]
                self.dss = np.concatenate(dss_matrix_list, axis=0).mean(0) if len(dss_matrix_list) >= 2 else\
                                                                                dss_matrix_list[0][0]

                if self.mss.shape[0] != self.mss.shape[-1] and self.mss.shape[1] == self.adj_matrix.shape[0]:
                    self.mss = self.mss.T

                if self.dss.shape[0] != self.dss.shape[-1] and self.dss.shape[1] == self.adj_matrix.shape[1]:
                    self.dss = self.dss.T

                assert self.mss.shape[0] == self.adj_matrix.shape[0]
                assert  self.dss.shape[0] == self.adj_matrix.shape[1]

                if self.mss.shape[0] != self.mss.shape[-1]:
                    self.mss = cosine_similarity(self.mss)

                if self.dss.shape[0] != self.dss.shape[-1]:
                    self.dss = cosine_similarity(self.dss)

                self.mss_Di = DiModel(self.kmeans_max_K, self.mss, self.kmeans_TopN)
                self.dss_Di = DiModel(self.kmeans_max_K, self.dss, self.kmeans_TopN)
                adj_matrix_mc = self.mss_Di.DiC
                adj_matrix_dc = self.dss_Di.DiC

                self.mss = np.concatenate([self.mss, np.ones((1, self.mss.shape[-1]))], axis=0)
                self.dss = np.concatenate([self.dss, np.ones((1, self.dss.shape[-1]))], axis=0)
                self.mss_graph = self.get_ss_edge(self.mss)
                self.dss_graph = self.get_ss_edge(self.dss)

                self.ASS_Embedding['edge_matrix'] = self.adj_matrix
                self.ASS_Embedding['miRNA_matrix'] = self.mss
                self.ASS_Embedding['Disease_matrix'] = self.dss

                adj_matrix = torch.tensor(self.adj_matrix)
                self.adj_matrix_torch = adj_matrix.clone().to(device)
                self.adj_matrix_mc = adj_matrix_mc = torch.tensor(adj_matrix_mc)
                self.adj_matrix_dc = adj_matrix_dc = torch.tensor(adj_matrix_dc)
                bm_numb = self.m_numb = adj_matrix.shape[0]
                bd_numb = self.d_numb = adj_matrix.shape[1]
                bcm_numb = self.cm_numb = adj_matrix_mc.shape[1]
                bcd_numb = self.cd_numb = adj_matrix_dc.shape[1]

                src, dst, etype = [], [], []
                self.node = {}

                bm = 0
                bd = bm + bm_numb
                temp = adj_matrix.nonzero()
                mids, dids = temp[:, 0], temp[:, 1]
                src += (bm + mids).tolist()
                dst += (bd + dids).tolist()
                etype += [self.num_rels] * (mids.shape[0])
                self.node[0] = ['miRNA', 0, self.m_numb, self.m_numb]
                self.node[1] = ['disease', self.node[0][-1], self.node[0][-1] + self.d_numb, self.d_numb]

                self.metapath_name[self.num_rels] = ('miRNA', self.num_rels, 'disease')
                self.num_rels += 1

                src += (bd + dids).tolist()
                dst += (bm + mids).tolist()
                etype += [self.num_rels] * (mids.shape[0])
                self.num_nodes = bd + bd_numb

                self.metapath_name[self.num_rels] = ('disease', self.num_rels, 'miRNA')

                self.num_rels += 1

                self.bcm = bcm = bd + bd_numb
                temp = adj_matrix_mc.nonzero()
                mids, cmids = temp[:, 0], temp[:, 1]
                src += (bm + mids).tolist()
                dst += (bcm + cmids).tolist()
                etype += [self.num_rels] * (mids.shape[0])

                self.metapath_name[self.num_rels] = ('miRNA', self.num_rels, 'DimiRNA')
                self.num_rels += 1

                src += (bcm + cmids).tolist()
                dst += (bm + mids).tolist()
                etype += [self.num_rels] * (mids.shape[0])
                self.num_nodes = bcm + bcm_numb
                self.node[2] = ['DimiRNA', self.node[1][-1], self.node[1][-1] + self.cm_numb, self.cm_numb]

                self.metapath_name[self.num_rels] = ('DimiRNA', self.num_rels, 'miRNA')
                self.num_rels += 1

                self.bcd = bcd = bcm + bcm_numb
                temp = adj_matrix_dc.nonzero()
                dids, cdids = temp[:, 0], temp[:, 1]
                src += (bd + dids).tolist()
                dst += (bcd + cdids).tolist()
                etype += [self.num_rels] * (dids.shape[0])
                self.metapath_name[self.num_rels] = ('disease', self.num_rels, 'Didisease')

                self.num_rels += 1

                src += (bcd + cdids).tolist()
                dst += (bd + dids).tolist()
                etype += [self.num_rels] * (dids.shape[0])
                self.num_nodes = bcd + bcd_numb

                self.node[3] = ['Didisease', self.node[2][-1], self.node[2][-1] + self.cd_numb, self.cd_numb]
                self.metapath_name[self.num_rels] = ('DimiRNA', self.num_rels, 'miRNA')

                self.num_nodes = self.num_nodes + 2

                self.src = src
                self.dst = dst
                self.etype = etype

                self.graph = dgl.graph((self.src, self.dst), num_nodes=self.num_nodes).to(self.device)
                self.graph.edata['type'] = torch.LongTensor(self.etype).to(self.device)
                self_loop_src = torch.arange(self.num_nodes).to(self.device)
                self_loop_dst = torch.arange(self.num_nodes).to(self.device)
                self.graph.add_edges(self_loop_src, self_loop_dst)
                neg_beta = self.config.neg_beta
                neg_beta = neg_beta + 1e-8
                self.x_encode, self.edge_and_label = get_y_data(self.adj_matrix, mm_list=m_encoding_list, dd_list=d_encoding_list,
                                                                x_encoding=x_encoding, neg_beta=neg_beta)
                self.graph.edata['old_type'] = self.graph.edata['type'].clone()
                self._predata = [self.adj_matrix,
                                 mss_list,
                                 dss_list,
                                 m_encoding_list,
                                 d_encoding_list,
                                 x_encoding]
                self.config.get_data_params(self)

    def resplit_Gdata(self,neg_beta=None):
        (md,
         mss_list,
         dss_list,
         m_encoding_list,
         d_encoding_list,
         x_encoding) =  self._predata
        if neg_beta is None:
            neg_beta = self.config.neg_beta
        neg_beta = neg_beta + 1e-8
        self.x_encode, self.edge_and_label = get_y_data(md, mm_list=m_encoding_list, dd_list=d_encoding_list, x_encoding=x_encoding, neg_beta=neg_beta)
        self.graph.edata['old_type'] = self.graph.edata['type'].clone()
        self.config.get_data_params(self)

    def add_new_metapath(self, metapath_adj_matrix, metapath_node1, metapath_node2, metapath_name=None, bidirectional=True):

        name1 = metapath_node1
        name2 = metapath_node2
        if type(metapath_node1) == str:
            for k,v in self.node.items():
                if v[0] == metapath_node1:
                    name1 = metapath_node1
                    metapath_node1=k
                    continue

        if type(metapath_node2) == str:
            for k,v in self.node.items():
                if v[0] == metapath_node1:
                    name2 = metapath_node2
                    metapath_node2 = k
                    continue

        assert metapath_node1 in self.node.keys() or metapath_node2 in self.node.keys()
        new_node_number = 2
        self.num_nodes = self.num_nodes-2
        if metapath_node1 in self.node.keys() and metapath_node2 in self.node.keys():
            assert metapath_adj_matrix.shape[0] == self.node[metapath_node1][-1]
            assert metapath_adj_matrix.shape[1] == self.node[metapath_node2][-1]
            temp = metapath_adj_matrix.nonzero()
            self.num_rels += 1
            sids, tids = temp[:, 0], temp[:, 1]
            self.src += (self.node[metapath_node1][1] + sids).tolist()
            self.dst += (self.node[metapath_node2][1] + tids).tolist()
            self.etype += [self.num_rels] * (sids.shape[0])
            if metapath_name != None:
                self.metapath_name[self.num_rels] = metapath_name

            if bidirectional==True:
                self.num_rels += 1
                self.src += (self.node[metapath_node2][1] + tids).tolist()
                self.dst += (self.node[metapath_node1][1] + sids).tolist()
                self.etype += [self.num_rels] * (sids.shape[0])
                if metapath_name != None:
                    self.metapath_name[self.num_rels] = metapath_name[::-1]

        elif metapath_node1 in self.node.keys():
            assert metapath_adj_matrix.shape[0] == self.node[metapath_node1][-1]
            mshape = metapath_adj_matrix.shape
            temp = metapath_adj_matrix.nonzero()
            self.num_rels += 1
            sids, tids = temp[:, 0], temp[:, 1]
            self.src += (self.node[metapath_node1][1] + sids).tolist()
            self.dst += (self.num_nodes + tids).tolist()
            self.etype += [self.num_rels] * (sids.shape[0])
            if metapath_name != None:
                self.metapath_name[self.num_rels] = metapath_name

            if bidirectional==True:
                self.num_rels += 1
                self.src += (self.num_nodes + tids).tolist()
                self.dst += (self.node[metapath_node1][1] + sids).tolist()
                self.etype += [self.num_rels] * (sids.shape[0])
                if metapath_name != None:
                    self.metapath_name[self.num_rels] = metapath_name[::-1]

            self.node[max(self.node.keys())+1] = [name2, self.num_nodes, self.num_nodes + mshape[1], mshape[1]]
            self.num_nodes += mshape[1]

            return

        elif metapath_node2 in self.node.keys():
            assert metapath_adj_matrix.shape[1] == self.node[metapath_node2][-1]
            mshape = metapath_adj_matrix.shape
            temp = metapath_adj_matrix.nonzero()
            self.num_rels += 1
            sids, tids = temp[:, 0], temp[:, 1]
            self.src += (self.num_nodes + sids).tolist()
            self.dst += (self.node[metapath_node1][1] + tids).tolist()
            self.etype += [self.num_rels] * (sids.shape[0])
            if metapath_name != None:
                self.metapath_name[self.num_rels] = metapath_name

            if bidirectional == True:
                self.num_rels += 1
                self.src += (self.node[metapath_node1][1] + tids).tolist()
                self.dst += (self.num_nodes + sids).tolist()
                self.etype += [self.num_rels] * (sids.shape[0])
                if metapath_name != None:
                    self.metapath_name[self.num_rels] = metapath_name[::-1]

            self.node[max(self.node.keys()) + 1] = [name1, self.num_nodes, self.num_nodes + mshape[0], mshape[0]]
            self.num_nodes += mshape[0]

        self.num_nodes = self.num_nodes + 2

        self.graph = dgl.graph((self.src, self.dst), num_nodes=self.num_nodes).to(self.device)
        self.graph.edata['type'] = torch.LongTensor(self.etype).to(self.device)

        self_loop_src = torch.arange(self.num_nodes).to(self.device)
        self_loop_dst = torch.arange(self.num_nodes).to(self.device)
        self.graph.add_edges(self_loop_src, self_loop_dst)
        self.graph.edata['old_type'] = self.graph.edata['type'].clone()

    def read_new_ss(self,mss_list=None, dss_list=None):
        mss_matrix_list = []
        dss_matrix_list = []
        if mss_matrix_list != None:
            for mss_f in mss_list:
                mss_matrix_list.append(pd.read_csv(mss_f, header=None, index_col=None).values[np.newaxis, ...])
            self.new_mss = np.concatenate(mss_matrix_list, axis=0).mean(0)
            if self.new_mss.shape[0] == self.mss.shape[-1]:
                self.new_mss = self.new_mss.T
            assert self.new_mss.shape[-1] == self.mss.shape[-1]

        if dss_list != None:
            for dss_f in dss_list:
                dss_matrix_list.append(pd.read_csv(dss_f, header=None, index_col=None).values[np.newaxis, ...])

            self.new_dss = np.concatenate(dss_matrix_list, axis=0).mean(0)

            if self.new_dss.shape[0] == self.dss.shape[-1]:
                self.new_dss = self.new_dss.T

            assert self.new_dss.shape[-1] == self.dss.shape[-1]

    def get_new_mss(self):
        for i, ss in enumerate(self.new_mss):
            self.mss_idx = i
            yield ss, i

    def get_new_dss(self):
        for i, ss in enumerate(self.new_dss):
            self.dss_idx = i
            yield ss, i

    def get_new_RNA_ass(self, idx=None):
        if idx==None:
            if self.mss_idx !=None:
                idx = self.mss_idx
            else:
                idx = 0
        return self.new_RNA_ass[idx]

    def get_new_Disease_ass(self, idx=None):
        if idx==None:
            if self.dss_idx !=None:
                idx = self.dss_idx
            else:
                idx = 0
        return self.new_Disease_ass[idx]

    def get_new_ass(self, midx=None, didx=None):
        if midx==None:
            if self.mss_idx !=None:
                midx = self.mss_idx
            else:
                midx = 0
        if didx==None:
            if self.dss_idx !=None:
                didx = self.dss_idx
            else:
                didx = 0
        return self.new_RNA_new_Disease_ass[midx, didx]

    def add_one_new_mss(self, new_mss=None):
        new_mss = new_mss.reshape((1, -1))
        assert new_mss.shape[-1] == self.mss.shape[-1]
        mss = self.mss.copy()
        mss[-1] = new_mss.reshape((1,-1))
        self.new_mss_graph = self.get_ss_edge(mss,new_ss=True)
        new_medges = torch.tensor(self.mss_Di.transform(new_mss))

        temp = new_medges.nonzero()
        mids, cmids = temp[:, 0], temp[:, 1]
        new_src = [self.num_nodes-2 for _ in range(mids.shape[0])]
        new_dst = (self.bcm + cmids).tolist()
        etype = [2] * mids.shape[0]

        new_src += (self.bcm + cmids).tolist()
        new_dst += [self.num_nodes-2 for _ in range(mids.shape[0])]
        etype += [3] * mids.shape[0]

        self.graph.add_edges(new_src, new_dst)
        self.graph.edata['type'][-len(etype):] = torch.tensor(etype).to(self.device)

    def add_one_new_dss(self, new_dss=None):
        new_dss = new_dss.reshape((1, -1))
        assert new_dss.shape[-1] == self.dss.shape[-1]
        dss = self.dss.copy()
        dss[-1] = new_dss.reshape((1,-1))
        self.new_dss_graph = self.get_ss_edge(dss, new_ss=True)
        new_dedges = torch.tensor(self.dss_Di.transform(new_dss))

        temp = new_dedges.nonzero()
        dids, cdids = temp[:, 0], temp[:, 1]
        new_src = [self.num_nodes-1 for _ in range(dids.shape[0])]
        new_dst = (self.bcd + cdids).tolist()
        etype = [4] * dids.shape[0]

        new_src += (self.bcm + cdids).tolist()
        new_dst += [self.num_nodes-1 for _ in range(dids.shape[0])]
        etype += [5] * dids.shape[0]

        self.graph.add_edges(new_src, new_dst)
        self.graph.edata['type'][-len(etype):] = torch.tensor(etype).to(self.device)

    def read_new_RNAassDisease(self, new_md_file=None):
        if new_md_file !=None:
            adj_matrix = pd.read_csv(new_md_file, header=None, index_col=None).values
            if adj_matrix.shape[0] == self.adj_matrix.shape[0]:
                adj_matrix = adj_matrix.T
            assert adj_matrix.shape[-1] == self.adj_matrix.shape[-1]
            assert adj_matrix.shape[0] == self.new_mss.shape[0]
            self.new_RNA_ass = adj_matrix

    def read_new_DiseaseassRNA(self, new_md_file=None):
        if new_md_file !=None:

            adj_matrix = pd.read_csv(new_md_file, header=None, index_col=None).values
            if adj_matrix.shape[0] == self.adj_matrix.shape[0]:
                adj_matrix = adj_matrix.T
            assert adj_matrix.shape[-1] == self.adj_matrix.shape[0]
            assert adj_matrix.shape[0] == self.new_dss.shape[0]
            self.new_Disease_ass = adj_matrix

    def read_new_RNA_new_Disease_ass(self, new_md_file=None):
        if new_md_file !=None:
            adj_matrix = pd.read_csv(new_md_file, header=None, index_col=None).values
            assert adj_matrix.shape[0] == self.new_mss.shape[0]
            assert adj_matrix.shape[1] == self.new_dss.shape[0]
            self.new_RNA_new_Disease_ass = adj_matrix

    def clear_newss(self):
        edge_ids_to_remove = []

        for node in [self.num_nodes-2, self.num_nodes-1]:
            succ = self.graph.successors(node)
            prede = self.graph.predecessors(node)
            for nei in succ:
                if node != nei:
                    edge_ids_to_remove.append(self.graph.edge_ids(node, nei))
            for nei in prede:
                if node != nei:
                    edge_ids_to_remove.append(self.graph.edge_ids(nei, node))

        if edge_ids_to_remove:
            edge_ids_to_remove = torch.cat(edge_ids_to_remove)
            self.graph = dgl.remove_edges(self.graph, edge_ids_to_remove)

        self.graph.edata['type'] = self.graph.edata['old_type'].clone()

    def get_ss_edge(self, similarity_matrix, TopK_ss=None, new_ss=False):
        if TopK_ss==None:
            TopK_ss=self.TopK_ss
        N, M = similarity_matrix.shape
        if new_ss == False:
            N = M
        src = []
        dst = []
        weights = []

        for i in range(N):
            similarities = similarity_matrix[i, :]
            if i < M:
                similarities[i] = -np.inf

            top_k_indices = np.argsort(similarities)[-(TopK_ss+1):][::-1]
            top_k_values = similarities[top_k_indices]

            for j, weight in zip(top_k_indices, top_k_values):
                src.append(i)
                dst.append(j)
                weights.append(weight)

        g = dgl.graph((src+dst,dst+src), num_nodes=similarity_matrix.shape[0]).to(self.device)
        g.edata['weight'] = torch.tensor(weights+weights, dtype=torch.float32).to(self.device)
        return g

    def copy(self):
        return copy.deepcopy(self)

    def save_ASS_Embedding(self, path):
        joblib.dump(self.ASS_Embedding, path)

    def load_ASS_Embedding(self, path):
        self.ASS_Embedding=joblib.load(path)
        return self.ASS_Embedding

    def set_rna_name(self, rna_name):
        self.rna_name = np.array(rna_name).tolist()

    def set_disease_name(self, disease_name):
        self.disease_name = np.array(disease_name).tolist()

    def get_top_mirnas_for_disease(
            self,
            given_disease_name=None,
            top_n=20
    ):

        mirna_names = self.rna_name
        disease_names = self.disease_name
        edge_scores = self.ASS_Embedding['EdgeScores']
        edge = self.ASS_Embedding['edge_matrix']

        if given_disease_name not in disease_names:
            raise ValueError(f"The disease name '{given_disease_name}' is not in the disease list.")

        disease_index = disease_names.index(given_disease_name)

        if edge_scores is None:
            raise ValueError("A valid score matrix needs to be provided.")

        column_scores = edge_scores[:, disease_index]
        top_indices = np.argsort(column_scores)[-top_n:][::-1]
        top_scores = column_scores[top_indices]

        column_scores = edge[:, disease_index]
        top_indices = np.argsort(column_scores)[-top_n:][::-1]
        top_edge = column_scores[top_indices]

        data = {
            'index': range(1, top_n + 1),
            'RNA name': [mirna_names[i] for i in top_indices],
            'Association score': top_scores,
            'Association edge': top_edge,
        }

        return pd.DataFrame(data)

    def get_top_mirnas_for_multiple_diseases(
            self,
            disease_list=None,
            top_n=20
    ):

        if not disease_list:
            raise ValueError("A list of disease names must be provided.")

        invalid_diseases = [d for d in disease_list if d not in self.disease_name]
        if invalid_diseases:
            raise ValueError(f"The following disease names are not in the list: {', '.join(invalid_diseases)}")

        all_results = []

        for disease in disease_list:
            disease_index = self.disease_name.index(disease)
            column_scores = self.ASS_Embedding['EdgeScores'][:, disease_index]

            top_indices = np.argsort(column_scores)[-top_n:][::-1]
            top_scores = column_scores[top_indices]

            column_scores = self.ASS_Embedding['edge_matrix'][:, disease_index]
            top_indices = np.argsort(column_scores)[-top_n:][::-1]
            top_edge = column_scores[top_indices]

            disease_df = pd.DataFrame({
                'index': range(1, top_n + 1),
                'RNA name': [self.rna_name[i] for i in top_indices],
                'Association score': top_scores,
                'Association edge': top_edge,
                'disease': disease                                   
            })

            all_results.append(disease_df)

        if not all_results:
            return pd.DataFrame()

        combined_df = pd.concat(all_results, ignore_index=True)

        combined_df = combined_df.set_index(['disease', 'index'])
        return combined_df

    def get_overlapping_disease_for_disease(
            self,
            given_disease_name=None,
            top_n=10,
            top_n_disease=5,
            min_score=0.5,
    ):

        gene_names = self.rna_name
        disease_names = self.disease_name
        edge_scores = self.ASS_Embedding['EdgeScores']
        edge_matrix = self.ASS_Embedding['edge_matrix']

        if given_disease_name not in disease_names:
            raise ValueError(f"The disease name '{given_disease_name}' is not in the disease list")

        disease_index = disease_names.index(given_disease_name)

        if edge_scores is None or edge_scores.size == 0:
            raise ValueError("A valid score matrix must be provided")
        if edge_matrix is None or edge_matrix.size == 0:
            raise ValueError("A valid edge matrix must be provided")

        column_scores = edge_scores[:, disease_index]
        top_indices = np.argsort(column_scores)[-top_n:][::-1]
        top_genes = [gene_names[i] for i in top_indices]
        top_scores = column_scores[top_indices]
        top_edge = edge_matrix[:, disease_index][top_indices]

        valid_main = np.where(top_scores >= min_score)[0]
        top_genes = [top_genes[i] for i in valid_main]
        top_scores = top_scores[valid_main]
        top_edge = top_edge[valid_main]

        spearman_corrs = {}
        all_other_diseases = [d for d in disease_names if d != given_disease_name]

        for disease in all_other_diseases:
            disease_idx = disease_names.index(disease)
            disease_scores = edge_scores[:, disease_idx]

            if len(disease_scores) >= top_n:
                corr, _ = spearmanr(column_scores, disease_scores)
                spearman_corrs[disease] = corr
            else:
                spearman_corrs[disease] = np.nan

        valid_diseases = [d for d in all_other_diseases if not np.isnan(spearman_corrs[d])]
        sorted_diseases = sorted(
            valid_diseases,
            key=lambda x: spearman_corrs[x],
            reverse=True
        )[:top_n_disease]

        all_results = []

        main_data = {
            'disease': ['Main: ' + given_disease_name] * len(top_genes),
            'index': range(1, len(top_genes) + 1),
            'RNA name': top_genes,
            'Association score': top_scores,
            'Association edge': top_edge
        }
        all_results.append(pd.DataFrame(main_data))

        for disease in sorted_diseases:
            disease_idx = disease_names.index(disease)
            disease_scores = edge_scores[:, disease_idx]
            disease_edge = edge_matrix[:, disease_idx]

            shared_genes = []
            shared_scores = []
            shared_edges = []

            for gene in top_genes:
                if gene in gene_names:
                    gene_idx = gene_names.index(gene)
                    if disease_scores[gene_idx] >= min_score:
                        shared_genes.append(gene)
                        shared_scores.append(disease_scores[gene_idx])
                        shared_edges.append(disease_edge[gene_idx])

            sorted_indices = np.argsort(shared_scores)[::-1]
            shared_genes = [shared_genes[i] for i in sorted_indices]
            shared_scores = [shared_scores[i] for i in sorted_indices]
            shared_edges = [shared_edges[i] for i in sorted_indices]

            disease_label = f"{disease} (spearman: {spearman_corrs[disease]:.4f})"

            disease_data = {
                'disease': [disease_label] * len(shared_genes),
                'index': range(1, len(shared_genes) + 1),
                'RNA name': shared_genes,
                'Association score': shared_scores,
                'Association edge': shared_edges
            }
            all_results.append(pd.DataFrame(disease_data))

        combined_df = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()

        if not combined_df.empty:
            combined_df = combined_df.set_index(['disease', 'index'])

        return combined_df
