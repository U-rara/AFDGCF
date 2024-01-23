import numpy as np
import torch
from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_uniform_initialization
from recbole.model.loss import BPRLoss, EmbLoss
from recbole.utils import InputType
from scipy.sparse import coo_matrix


class GCCF(GeneralRecommender):
    r"""LightGCN is a GCN-based recommender model.
    LightGCN includes only the most essential component in GCN — neighborhood aggregation — for
    collaborative filtering. Specifically, LightGCN learns user and item embeddings by linearly
    propagating them on the user-item interaction graph, and uses the weighted sum of the embeddings
    learned at all layers as the final embedding.
    We implement the model following the original author with a pairwise training mode.
    """
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(GCCF, self).__init__(config, dataset)

        # load dataset info
        self.interaction_matrix = dataset.inter_matrix(form="coo").astype(np.float32)

        self.user_degree = np.array(self.interaction_matrix.sum(axis=1)).flatten()
        self.item_degree = np.array(self.interaction_matrix.sum(axis=0)).flatten()
        self.user_degree_inverse = 1 / (self.user_degree + 1)
        self.item_degree_inverse = 1 / (self.item_degree + 1)

        self.norm_interaction_matrix = self.get_norm_interaction_matrix()
        self.user_item_matrix = self.coo_to_pytorch_sparse(self.norm_interaction_matrix).to(self.device)
        self.item_user_matrix = self.coo_to_pytorch_sparse(self.norm_interaction_matrix.T).to(self.device)

        # load parameters info
        self.latent_dim = config["embedding_size"]  # int type:the embedding size of lightGCN
        self.n_layers = config["n_layers"]  # int type:the layer num of lightGCN
        self.reg_weight = config["reg_weight"]  # float32 type: the weight decay for l2 normalization
        self.require_pow = config["require_pow"]

        # define layers and loss
        self.user_embedding = torch.nn.Embedding(
            num_embeddings=self.n_users, embedding_dim=self.latent_dim
        )
        self.item_embedding = torch.nn.Embedding(
            num_embeddings=self.n_items, embedding_dim=self.latent_dim
        )
        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()

        # storage variables for full sort evaluation acceleration
        self.restore_user_e = None
        self.restore_item_e = None

        self.d_i_train = torch.from_numpy(self.user_degree_inverse).unsqueeze(1).expand(-1, self.latent_dim).to(
            self.device)
        self.d_j_train = torch.from_numpy(self.item_degree_inverse).unsqueeze(1).expand(-1, self.latent_dim).to(
            self.device)

        # parameters initialization
        self.apply(xavier_uniform_initialization)
        self.other_parameter_name = ["restore_user_e", "restore_item_e"]

    def get_norm_interaction_matrix(self):
        nonzero_indices = np.transpose(self.interaction_matrix.nonzero())
        values = np.sqrt(
            self.user_degree_inverse[nonzero_indices[:, 0]] * self.item_degree_inverse[nonzero_indices[:, 1]])
        return coo_matrix((values, (nonzero_indices[:, 0], nonzero_indices[:, 1])), shape=self.interaction_matrix.shape)

    def coo_to_pytorch_sparse(self, coo_sparse_matrix):
        coo_rows = coo_sparse_matrix.row
        coo_cols = coo_sparse_matrix.col
        coo_vals = coo_sparse_matrix.data

        sparse_tensor = torch.sparse_coo_tensor(
            torch.tensor([coo_rows, coo_cols]),
            torch.tensor(coo_vals),
            torch.Size(coo_sparse_matrix.shape)
        )

        return sparse_tensor

    def forward(self, require_embeddings_list=False):
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        gcn_user_embeddings_list = [user_embeddings]
        gcn_item_embeddings_list = [item_embeddings]
        gcn_all_embeddings_list = [torch.cat([user_embeddings, item_embeddings], dim=0)]

        for layer in range(self.n_layers):
            gcn_user = torch.sparse.mm(self.user_item_matrix, gcn_item_embeddings_list[-1]) + \
                       gcn_user_embeddings_list[-1].mul(self.d_i_train)

            gcn_item = torch.sparse.mm(self.item_user_matrix, gcn_user_embeddings_list[-1]) + \
                       gcn_item_embeddings_list[-1].mul(self.d_j_train)

            gcn_user_embeddings_list.append(gcn_user)
            gcn_item_embeddings_list.append(gcn_item)
            gcn_all_embeddings_list.append(torch.cat([gcn_user, gcn_item], dim=0))

        gcn_user_embeddings = torch.cat(gcn_user_embeddings_list, -1)
        gcn_item_embeddings = torch.cat(gcn_item_embeddings_list, -1)

        if require_embeddings_list:
            return gcn_user_embeddings, gcn_item_embeddings, gcn_all_embeddings_list
        else:
            return gcn_user_embeddings, gcn_item_embeddings

    def calculate_loss(self, interaction, batch_idx=None):
        # clear the storage variable when training
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        user_all_embeddings, item_all_embeddings = self.forward()
        u_embeddings = user_all_embeddings[user]
        pos_embeddings = item_all_embeddings[pos_item]
        neg_embeddings = item_all_embeddings[neg_item]

        # calculate BPR Loss
        pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)
        mf_loss = self.mf_loss(pos_scores, neg_scores)

        # calculate BPR Loss
        u_ego_embeddings = self.user_embedding(user)
        pos_ego_embeddings = self.item_embedding(pos_item)
        neg_ego_embeddings = self.item_embedding(neg_item)

        reg_loss = self.reg_loss(
            u_ego_embeddings,
            pos_ego_embeddings,
            neg_ego_embeddings,
            require_pow=self.require_pow,
        )

        loss = mf_loss + self.reg_weight * reg_loss

        return loss

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
