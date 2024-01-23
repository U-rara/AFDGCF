import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_normal_initialization
from recbole.model.layers import BiGNNLayer, SparseDropout
from recbole.model.loss import BPRLoss, EmbLoss
from recbole.utils import InputType

from loss import calculate_correlation


class AFD_NGCF(GeneralRecommender):
    r"""NGCF is a model that incorporate GNN for recommendation.
    We implement the model following the original author with a pairwise training mode.
    """
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(AFD_NGCF, self).__init__(config, dataset)

        # load dataset info
        self.interaction_matrix = dataset.inter_matrix(form="coo").astype(np.float32)

        # load parameters info
        self.embedding_size = config["embedding_size"]
        self.hidden_size_list = config["hidden_size_list"]
        self.hidden_size_list = [self.embedding_size] + self.hidden_size_list
        self.node_dropout = config["node_dropout"]
        self.message_dropout = config["message_dropout"]
        self.reg_weight = config["reg_weight"]

        self.alpha = config["alpha"]

        # define layers and loss
        self.sparse_dropout = SparseDropout(self.node_dropout)
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size)
        self.GNNlayers = torch.nn.ModuleList()
        for idx, (input_size, output_size) in enumerate(
                zip(self.hidden_size_list[:-1], self.hidden_size_list[1:])
        ):
            self.GNNlayers.append(BiGNNLayer(input_size, output_size))
        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()

        # storage variables for full sort evaluation acceleration
        self.restore_user_e = None
        self.restore_item_e = None

        # generate intermediate data
        self.norm_adj_matrix = self.get_norm_adj_mat().to(self.device)
        self.eye_matrix = self.get_eye_mat().to(self.device)

        # parameters initialization
        self.apply(xavier_normal_initialization)
        self.other_parameter_name = ["restore_user_e", "restore_item_e"]

    def get_norm_adj_mat(self):
        r"""Get the normalized interaction matrix of users and items.
        Construct the square matrix from the training data and normalize it
        using the laplace matrix.
        .. math::
            A_{hat} = D^{-0.5} \times A \times D^{-0.5}
        Returns:
            Sparse tensor of the normalized interaction matrix.
        """
        # build adj matrix
        A = sp.dok_matrix(
            (self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32
        )
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(
            zip(zip(inter_M.row, inter_M.col + self.n_users), [1] * inter_M.nnz)
        )
        data_dict.update(
            dict(
                zip(
                    zip(inter_M_t.row + self.n_users, inter_M_t.col),
                    [1] * inter_M_t.nnz,
                )
            )
        )
        A._update(data_dict)
        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        diag = (
                np.array(sumArr.flatten())[0] + 1e-7
        )  # add epsilon to avoid divide by zero Warning
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse.FloatTensor(i, data, torch.Size(L.shape))
        return SparseL

    def get_eye_mat(self):
        r"""Construct the identity matrix with the size of  n_items+n_users.
        Returns:
            Sparse tensor of the identity matrix. Shape of (n_items+n_users, n_items+n_users)
        """
        num = self.n_items + self.n_users  # number of column of the square matrix
        i = torch.LongTensor([range(0, num), range(0, num)])
        val = torch.FloatTensor([1] * num)  # identity matrix
        return torch.sparse.FloatTensor(i, val)

    def get_ego_embeddings(self):
        r"""Get the embedding of users and items and combine to an embedding matrix.
        Returns:
            Tensor of the embedding matrix. Shape of (n_items+n_users, embedding_dim)
        """
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings

    def forward(self, require_embeddings_list=False):

        A_hat = (
            self.sparse_dropout(self.norm_adj_matrix)
            if self.node_dropout != 0
            else self.norm_adj_matrix
        )
        all_embeddings = self.get_ego_embeddings()
        embeddings_list = [all_embeddings]
        for gnn in self.GNNlayers:
            all_embeddings = gnn(A_hat, self.eye_matrix, all_embeddings)
            all_embeddings = nn.LeakyReLU(negative_slope=0.2)(all_embeddings)
            all_embeddings = nn.Dropout(self.message_dropout)(all_embeddings)
            all_embeddings = F.normalize(all_embeddings, p=2, dim=1)
            embeddings_list += [
                all_embeddings
            ]  # storage output embedding of each layer
        ngcf_all_embeddings = torch.cat(embeddings_list, dim=1)

        user_all_embeddings, item_all_embeddings = torch.split(
            ngcf_all_embeddings, [self.n_users, self.n_items]
        )
        if require_embeddings_list:
            return user_all_embeddings, item_all_embeddings, embeddings_list
        else:
            return user_all_embeddings, item_all_embeddings

    def calculate_loss(self, interaction, batch_idx=None):
        # clear the storage variable when training
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        user_all_embeddings, item_all_embeddings, embeddings_list = self.forward(True)
        u_embeddings = user_all_embeddings[user]
        pos_embeddings = item_all_embeddings[pos_item]
        neg_embeddings = item_all_embeddings[neg_item]

        pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)
        mf_loss = self.mf_loss(pos_scores, neg_scores)  # calculate BPR Loss

        reg_loss = self.reg_loss(
            u_embeddings, pos_embeddings, neg_embeddings
        )  # L2 regularization of embeddings

        rec_loss = mf_loss + self.reg_weight * reg_loss

        cor_loss_u, cor_loss_i = torch.zeros((1,)).to(self.device), torch.zeros((1,)).to(self.device)

        user_layer_correlations = []
        item_layer_correlations = []
        for embeddings in embeddings_list[1:]:
            user_embeddings, item_embeddings = torch.split(embeddings, [self.n_users, self.n_items])
            user_layer_correlations.append(calculate_correlation(user_embeddings))
            item_layer_correlations.append(calculate_correlation(item_embeddings))

        user_layer_correlations_coef = (1 / torch.tensor(user_layer_correlations)) / torch.sum(
            1 / torch.tensor(user_layer_correlations))
        item_layer_correlations_coef = (1 / torch.tensor(item_layer_correlations)) / torch.sum(
            1 / torch.tensor(item_layer_correlations))

        for i in range(1, len(self.hidden_size_list)):
            cor_loss_u += user_layer_correlations_coef[i - 1] * user_layer_correlations[i - 1]
            cor_loss_i += item_layer_correlations_coef[i - 1] * item_layer_correlations[i - 1]

        return rec_loss, self.alpha * cor_loss_u, self.alpha * cor_loss_i

    def predict(self, interaction):

        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        user_all_embeddings, item_all_embeddings = self.forward()

        u_embeddings = user_all_embeddings[user]
        i_embeddings = item_all_embeddings[item]
        scores = torch.mul(u_embeddings, i_embeddings).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e = self.forward()
        # get user embedding from storage variable
        u_embeddings = self.restore_user_e[user]

        # dot with all item embedding to accelerate
        scores = torch.matmul(u_embeddings, self.restore_item_e.transpose(0, 1))

        return scores.view(-1)
