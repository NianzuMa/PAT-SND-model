import copy

from torch_geometric.nn import GATConv
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def mask_logits(target, mask):
    return target * mask + (1 - mask) * (-1e30)


def glorot(tensor):
    stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
    if tensor is not None:
        tensor.data.uniform_(-stdv, stdv)


class Highway(nn.Module):
    def __init__(self, layer_num, dim):
        super().__init__()
        self.layer_num = layer_num
        self.linear = nn.ModuleList([nn.Linear(dim, dim)
                                     for _ in range(layer_num)])
        self.gate = nn.ModuleList([nn.Linear(dim, dim)
                                   for _ in range(layer_num)])

    def forward(self, x):
        for i in range(self.layer_num):
            gate = F.sigmoid(self.gate[i](x))
            nonlinear = F.relu(self.linear[i](x))
            x = gate * nonlinear + (1 - gate) * x
        return x


# class RelationAttention(nn.Module):
#     def __init__(self, in_dim=300, hidden_dim=64):
#         # in_dim: the dimension of query vector
#         super().__init__()
#
#         self.fc1 = nn.Linear(in_dim, hidden_dim)
#         self.relu = nn.ReLU()
#         self.fc2 = nn.Linear(hidden_dim, 1)
#
#     def forward(self, feature, dep_tags_v, dmask):
#         '''
#         C feature/context [N, L, D]
#         Q dep_tags_v          [N, L, D]
#         mask dmask          [N, L]
#         '''
#         Q = self.fc1(dep_tags_v)  # dep_tags_v=(16,36,768), feature=(16,36,768) -> Q (16,36,64)
#         Q = self.relu(Q)
#         Q = self.fc2(Q)  # (N, L, 1)  -> (16,36,1)
#         Q = Q.squeeze(2)  # after squeeze -> (16,36)
#         Q = F.softmax(mask_logits(Q, dmask), dim=1)  # -> (16,36)
#
#         Q = Q.unsqueeze(2)  # (16, 36, 1)
#         out = torch.bmm(feature.transpose(1, 2), Q)
#         # torch.bmm Performs a batch matrix-matrix product of matrices stored in input and mat2.
#         # feature.transpose(1, 2) -> (16, 768, 36)
#         # Q -> (16, 36, 1)
#         # out -> (16, 768, 1)
#
#         out = out.squeeze(2)  # (16, 768)
#         # out = F.sigmoid(out)
#         return out  # ([N, L])


class RelationAttention(nn.Module):
    def __init__(self, num_of_rc_relation, in_dim=300, hidden_dim=64):
        # in_dim: the dimension of query vector
        super().__init__()

        # self.fc1 = nn.Linear(in_dim, hidden_dim)  # -> (768, 64)
        self.relu = nn.ReLU()

        # self.fc2_dict = {}
        # for i in range(num_of_rc_relation):
        #     self.fc2_dict[i] = nn.Linear(in_dim, 1)  # -> (768, 1)
        # # endfor

        # #### new #####
        linear_list = [nn.Linear(in_dim, 1) for i in range(num_of_rc_relation)]
        # initialize
        for layer in linear_list:
            glorot(layer.weight)
        #endfor
        self.fc2_list = torch.nn.ModuleList(linear_list)

    def forward(self, value_feature, property_feature, rc_relation_id):
        '''
        feature -> value_feature
        dep_tags_v -> property_feature

        C feature/context [N, L, D]
        Q dep_tags_v          [N, L, D]
        mask dmask          [N, L]
        '''

        # Q = self.fc1(dep_tags_v)  # dep_tags_v=(5,768), feature=(5,768) -> Q (5,64)
        Q = self.fc2_list[rc_relation_id](
            property_feature)  # property_feature: (10, 768),  fc2_list (768, 1) -> (10, 1)
        Q = self.relu(Q)
        Q = Q.squeeze(1)  # after squeeze -> (10,)
        Q = F.softmax(Q, dim=0)  # -> (10,)

        # attention_prob = copy.deepcopy(Q) # only object explicitly created by user can use deep.copy()
        attention_prob = Q.detach().cpu().numpy()

        Q = Q.unsqueeze(1)  # (10, 1)
        out = torch.matmul(value_feature.transpose(0, 1), Q)  # (768,10) x (10,1) -> (768,1)

        # out = torch.bmm(feature.transpose(0, 1), Q)  # (768,5) x (5,1) -> (768,1)
        # torch.bmm Performs a batch matrix-matrix product of matrices stored in input and mat2.
        # feature.transpose(1, 2) -> (16, 768, 36)
        # Q -> (16, 36, 1)
        # out -> (16, 768, 1)

        out = out.squeeze(1)  # (768,)
        # out = F.sigmoid(out)
        return out, attention_prob  # ([N, L])


class RelationAttention_with_Attention_Info(nn.Module):
    """
    To extract and return the attention information for interpretation
    """

    def __init__(self, in_dim=300, hidden_dim=64):
        # in_dim: the dimension of query vector
        super().__init__()

        # self.fc1 = nn.Linear(in_dim, hidden_dim)  # -> (768, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_dim, 1)  # -> (768, 1)

    def forward(self, feature, dep_tags_v):
        '''
        C feature/context [N, L, D]
        Q dep_tags_v          [N, L, D]
        mask dmask          [N, L]
        '''
        # print(">>>>>>>>>>> different attention")
        # Q = self.fc1(dep_tags_v)  # dep_tags_v=(5,768), feature=(5,768) -> Q (5,64)
        Q = self.fc2(dep_tags_v)  # (N, L, 1)  -> (5,1)
        Q = self.relu(Q)
        Q = Q.squeeze(1)  # after squeeze -> (5,)
        Q = F.softmax(Q, dim=0)  # -> (5,)

        # attention_prob = copy.deepcopy(Q) # only object explicitly created by user can use deep.copy()
        attention_prob = Q.detach().cpu().numpy()

        Q = Q.unsqueeze(1)  # (5, 1)
        out = torch.matmul(feature.transpose(0, 1), Q)  # (768,5) x (5,1) -> (768,1)

        # out = torch.bmm(feature.transpose(0, 1), Q)  # (768,5) x (5,1) -> (768,1)
        # torch.bmm Performs a batch matrix-matrix product of matrices stored in input and mat2.
        # feature.transpose(1, 2) -> (16, 768, 36)
        # Q -> (16, 36, 1)
        # out -> (16, 768, 1)

        out = out.squeeze(1)  # (768,)
        # out = F.sigmoid(out)
        return out, attention_prob  # ([N, L])


class GAT_MaxMargin_1(nn.Module):
    """
    GAT module operated on graphs
    """

    def __init__(self, args, property_embed_matrix_tensor, num_of_rc_relation, in_dim=768, hidden_size=64, mem_dim=300,
                 num_layers=1):
        super(GAT_MaxMargin_1, self).__init__()
        self.args = args
        self.property_embed_matrix_tensor = property_embed_matrix_tensor
        self.num_of_rc_relation = num_of_rc_relation
        self.num_layers = num_layers
        self.in_dim = in_dim
        self.dropout = nn.Dropout(args.gcn_dropout)
        self.leakyrelu = nn.LeakyReLU(1e-2)

        # Standard GAT:attention over feature
        a_layers = [
            nn.Linear(2 * mem_dim, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, 1)]

        self.afcs = nn.Sequential(*a_layers)

        # gcn layer
        self.W = nn.ModuleList()
        for layer in range(num_layers):
            input_dim = self.in_dim if layer == 0 else mem_dim
            self.W.append(nn.Linear(input_dim, mem_dim))
        # endfor

    def forward(self, adj, feature):
        B, N = adj.size(0), adj.size(1)
        dmask = adj.view(B, -1)  # (batch_size, n*n)
        # gcn layer
        for l in range(self.num_layers):
            # Standard GAT:attention over feature
            #####################################
            h = self.W[l](feature)  # (B, N, D)
            a_input = torch.cat([h.repeat(1, 1, N).view(
                B, N * N, -1), h.repeat(1, N, 1)], dim=2)  # (B, N*N, 2*D)
            e = self.leakyrelu(self.afcs(a_input)).squeeze(2)  # (B, N*N)
            attention = F.softmax(mask_logits(e, dmask), dim=1)
            attention = attention.view(*adj.size())

            # original gat
            feature = attention.bmm(h)
            feature = self.dropout(feature) if l < self.num_layers - 1 else feature
            #####################################

        return feature


class GAT_MaxMargin(nn.Module):
    """
    GAT module operated on graphs
    """

    # make the relation embedding trainable

    def __init__(self, args, property_embed_tensor, num_of_rc_relation, in_dim=768, hidden_size=64, mem_dim=300,
                 num_layers=1):
        super(GAT_MaxMargin, self).__init__()
        self.args = args

        self.property_embed_tensor = property_embed_tensor

        self.num_of_rc_relation = num_of_rc_relation

        # self.num_layers = num_layers

        self.in_dim = in_dim

        # self.dropout = nn.Dropout(args.gcn_dropout)

        self.leakyrelu = nn.LeakyReLU(1e-2)

        # Standard GAT:attention over feature
        # a_layers = [
        #     nn.Linear(2 * mem_dim, hidden_size), nn.ReLU(),
        #     nn.Linear(hidden_size, 1)]
        #
        # self.afcs = nn.Sequential(*a_layers)

        # gcn layer
        self.W = nn.ModuleList()
        for layer in range(num_layers):
            input_dim = self.in_dim if layer == 0 else mem_dim
            self.W.append(nn.Linear(input_dim, mem_dim))
        # endfor

        # ##### (1) transform W #########
        transform_linear_list = [nn.Linear(in_dim, hidden_size) for i in range(num_of_rc_relation)]
        # initialize
        for layer in transform_linear_list:
            glorot(layer.weight)
        # endfor
        self.value_transform_list = torch.nn.ModuleList(transform_linear_list)

        # ##### relation rc embedding ##### (trainable)
        self.property_embed = nn.Embedding.from_pretrained(property_embed_tensor, freeze=False)

        # ###### (2) layer a, the single layer feedforward neural network ######
        layer_a_list = [nn.Linear(2 * hidden_size, 1) for i in range(num_of_rc_relation)]
        # initialize
        for layer in layer_a_list:
            glorot(layer.weight)
        # endfor
        self.layer_a_list = torch.nn.ModuleList(layer_a_list)

        # ###### (3) out list ######
        out_layer_list = [nn.Linear(2 * in_dim, 1) for i in range(num_of_rc_relation)]
        # initialize
        for layer in out_layer_list:
            glorot(layer.weight)
        # endfor
        self.out_layer_list = torch.nn.ModuleList(out_layer_list)


    def forward(self,
                args,
                head_property_id_tensor_list,
                tail_property_id_tensor_list,
                head_property_value_embeds_matrix_tensor_list,
                tail_property_value_embeds_matrix_tensor_list,
                nd_label_id_list,
                rc_label_id_list):

        batch_size = len(head_property_id_tensor_list)

        total_score = []
        for i in range(len(rc_label_id_list)):
            # relation
            rc_label_id = rc_label_id_list[i]
            rc_label_id_tensor = torch.LongTensor([rc_label_id]).to(args.device)
            rc_relation_embed = self.property_embed(rc_label_id_tensor)  # size (1, 768)
            rc_relation_embed = torch.squeeze(rc_relation_embed, dim=0)
            rc_relation_embed_transformed = self.value_transform_list[rc_label_id](rc_relation_embed)  # size (64)

            # ######### head ##########
            head_property_value_embeds_matrix = head_property_value_embeds_matrix_tensor_list[i]  # (8, 768)
            head_value_size = head_property_value_embeds_matrix.size()[0]  # 

            h_e_list = []
            for h_i in range(head_value_size):
                transformed_head_value_entry = self.value_transform_list[rc_label_id](
                    head_property_value_embeds_matrix[h_i])  # size (64)
                head_cat = torch.cat([transformed_head_value_entry, rc_relation_embed_transformed], dim=0)
                head_score = self.layer_a_list[rc_label_id](head_cat)
                h_e = self.leakyrelu(head_score)
                h_e_list.append(h_e)
            # endfor
            h_e_arr = torch.stack(h_e_list)
            h_attention = F.softmax(h_e_arr, dim=0)  # (9, 1)
            head_property_value_embeds_matrix_prime = torch.transpose(head_property_value_embeds_matrix, 0,
                                                                      1)  # (9, 768) -> (768, 9)
            h_feature_final = torch.matmul(head_property_value_embeds_matrix_prime, h_attention)  # (768, 9) x (9, 1)
            h_feature_final = torch.squeeze(h_feature_final, dim=1)  # (768 x 1) -> (768)

            # ########### tail ##########
            tail_property_value_embeds_matrix = tail_property_value_embeds_matrix_tensor_list[i]  # (8, 768)
            tail_value_size = tail_property_value_embeds_matrix.size()[0]  # 

            t_e_list = []
            for t_i in range(tail_value_size):
                transformed_tail_value_entry = self.value_transform_list[rc_label_id](
                    tail_property_value_embeds_matrix[t_i])  # size (64)
                tail_cat = torch.cat([transformed_tail_value_entry, rc_relation_embed_transformed], dim=0)
                tail_score = self.layer_a_list[rc_label_id](tail_cat)
                t_e = self.leakyrelu(tail_score)
                t_e_list.append(t_e)
            # endfor
            t_e_arr = torch.stack(t_e_list)
            t_attention = F.softmax(t_e_arr, dim=0)  # (9, 1)
            tail_property_value_embeds_matrix_prime = torch.transpose(tail_property_value_embeds_matrix, 0,
                                                                      1)  # (9, 768) -> (768, 9)
            t_feature_final = torch.matmul(tail_property_value_embeds_matrix_prime, t_attention)  # (768, 9) x (9, 1)
            t_feature_final = torch.squeeze(t_feature_final, dim=1)  # (768 x 1) -> (768)

            # concatenate
            cat_out = torch.cat([h_feature_final, t_feature_final], dim=0)
            cat_out = torch.unsqueeze(cat_out, dim=0)
            score_out = self.out_layer_list[rc_label_id](cat_out)

            total_score.append(score_out)
        # endfor

        total_score = torch.cat(total_score, dim=1)
        total_score = torch.squeeze(total_score, dim=0)

        loss = None
        if nd_label_id_list is not None:
            # Note: get positive / negative logits here
            # positive
            pos_mask = nd_label_id_list.eq(1)
            pos_score = total_score[pos_mask]
            # negative
            neg_mask = nd_label_id_list.eq(0)
            neg_score = total_score[neg_mask]

            # Note: max margin loss
            max_margin_loss = nn.MarginRankingLoss(margin=1)
            target = torch.tensor([1] * int(batch_size / 2), dtype=torch.long).to(args.device)
            loss = max_margin_loss(pos_score, neg_score, target)
        # endif

        return total_score, loss


class RGAT(nn.Module):
    """
    relation-aware GAT
    """

    def __init__(self, args, property_embed_tensor, num_of_rc_relation):
        """
        property_num: the total number of properties involved in one-hop of all our entities
        """
        super(RGAT, self).__init__()
        self.args = args

        args.embedding_dim = 768
        # num_embeddings, embed_dim = args.glove_embedding.shape
        # self.embed = nn.Embedding(num_embeddings, embed_dim)
        # self.embed.weight = nn.Parameter(
        #     args.glove_embedding, requires_grad=False)

        self.dropout = nn.Dropout(args.dropout)
        self.tanh = nn.Tanh()

        # if args.highway:
        #     self.highway_dep = Highway(args.num_layers, args.embedding_dim)
        #     self.highway = Highway(args.num_layers, args.embedding_dim)

        # self.bilstm = nn.LSTM(input_size=args.embedding_dim, hidden_size=args.hidden_size,
        #                       bidirectional=True, batch_first=True, num_layers=args.num_layers)
        # gcn_input_dim = args.hidden_size * 2

        # if args.gat:
        args.num_heads = 8
        self.gat_dep = [
            RelationAttention(num_of_rc_relation=num_of_rc_relation, in_dim=args.embedding_dim).to(args.device) for i in
            range(args.num_heads)]

        # if args.gat_attention_type == 'linear':
        #     self.gat = [LinearAttention(in_dim = gcn_input_dim, mem_dim = gcn_input_dim).to(args.device) for i in range(args.num_heads)] # we prefer to keep the dimension unchanged
        # elif args.gat_attention_type == 'dotprod':
        #     self.gat = [DotprodAttention().to(args.device) for i in range(args.num_heads)]
        # else:
        #     # reshaped gcn
        #     self.gat = nn.Linear(gcn_input_dim, gcn_input_dim)

        # property_embedding
        # self.property_embed = nn.Embedding(property_num, args.embedding_dim)
        # does not fine-tune during training, freeze=True

        self.property_embed = nn.Embedding.from_pretrained(property_embed_tensor, freeze=False)

        # tmp_embed = nn.Embedding(property_embed.size()[0], property_embed.size()[1])
        # nn.init.uniform_(tmp_embed.weight, -1.0, 1.0)
        # self.property_embed = tmp_embed

        # model.embedding.weight.grad
        # torch.manual_seed(3)
        # emb2 = nn.Embedding(5, 5)
        # nn.init.uniform_(emb2.weight, -1.0, 1.0)

        # last_hidden_size = args.hidden_size * 4
        #
        # layers = [
        #     nn.Linear(last_hidden_size, args.final_hidden_size), nn.ReLU()]
        # for _ in range(args.num_mlps-1):
        #     layers += [nn.Linear(args.final_hidden_size,
        #                          args.final_hidden_size), nn.ReLU()]
        # self.fcs = nn.Sequential(*layers)
        # self.fc_final = nn.Linear(args.final_hidden_size, args.num_classes)
        self.lin1 = nn.Linear(768 * 2, 2)
        glorot(self.lin1.weight)
        pass

    def forward(self,
                args,
                head_property_id_tensor_list,
                tail_property_id_tensor_list,
                head_property_value_embeds_matrix_tensor_list,
                tail_property_value_embeds_matrix_tensor_list,
                nd_label_id_list,
                rc_label_id_list):
        '''
        Forward takes:
            sentence: sentence_id of size (batch_size, text_length)
            aspect: aspect_id of size (batch_size, aspect_length)
            pos_class: pos_tag_id of size (batch_size, text_length)
            dep_tags: dep_tag_id of size (batch_size, text_length)
            text_len: (batch_size,) length of each sentence
            aspect_len: (batch_size, ) aspect length of each sentence
            dep_rels: (batch_size, text_length) relation
            dep_heads: (batch_size, text_length) which node adjacent to that node
            aspect_position: (batch_size, text_length) mask, with the position of aspect as 1 and others as 0
            dep_dirs: (batch_size, text_length) the directions each node to the aspect
        '''
        # fmask = (torch.zeros_like(sentence) != sentence).float()  # (N，L)
        # dmask = (torch.zeros_like(dep_tags) != dep_tags).float()  # (N ,L)

        logits_list = []
        for i in range(len(rc_label_id_list)):
            # relation_embed = self.property_embed(target_relation_id)
            rc_label_id = rc_label_id_list[i]

            head_property_feature = self.property_embed(head_property_id_tensor_list[i])
            tail_property_feature = self.property_embed(tail_property_id_tensor_list[i])

            head_value_feature = head_property_value_embeds_matrix_tensor_list[i]
            tail_value_feature = tail_property_value_embeds_matrix_tensor_list[i]

            # here, it uses [0], to only use the first return value
            # (N, 1, D) * num_heads
            head_dep_out = [g(head_value_feature, head_property_feature, rc_label_id)[0].unsqueeze(1) for g in
                            self.gat_dep]
            # (N, 1, D) * num_heads 
            tail_dep_out = [g(tail_value_feature, tail_property_feature, rc_label_id)[0].unsqueeze(1) for g in
                            self.gat_dep]

            head_dep_out = torch.cat(head_dep_out, dim=1)  # (N, H, D)
            head_dep_out = head_dep_out.mean(dim=1)  # (768,)

            tail_dep_out = torch.cat(tail_dep_out, dim=1)
            tail_dep_out = tail_dep_out.mean(dim=1)  # (768,)

            cat_out = torch.cat([head_dep_out, tail_dep_out], dim=0)

            lin1_result = self.lin1(cat_out)
            # logit = torch.relu(lin1_result)
            logits_list.append(lin1_result)
        # endfor

        all_logits = torch.stack(logits_list)  # (batch_size, 2)

        if nd_label_id_list is not None:
            loss_func = nn.CrossEntropyLoss()
            loss = loss_func(all_logits, nd_label_id_list)
            return all_logits, loss
        else:
            return all_logits, None


class RGAT_with_Attention_Info(nn.Module):
    """
    relation-aware GAT
    """

    def __init__(self, args, property_num, property_embed):
        """
        property_num: the total number of properties involved in one-hop of all our entities
        """
        super(RGAT_with_Attention_Info, self).__init__()
        self.args = args

        args.embedding_dim = 768
        # num_embeddings, embed_dim = args.glove_embedding.shape
        # self.embed = nn.Embedding(num_embeddings, embed_dim)
        # self.embed.weight = nn.Parameter(
        #     args.glove_embedding, requires_grad=False)

        self.dropout = nn.Dropout(args.dropout)
        self.tanh = nn.Tanh()

        # if args.highway:
        #     self.highway_dep = Highway(args.num_layers, args.embedding_dim)
        #     self.highway = Highway(args.num_layers, args.embedding_dim)

        # self.bilstm = nn.LSTM(input_size=args.embedding_dim, hidden_size=args.hidden_size,
        #                       bidirectional=True, batch_first=True, num_layers=args.num_layers)
        # gcn_input_dim = args.hidden_size * 2

        # if args.gat:
        args.num_heads = 8
        self.gat_dep = [RelationAttention_with_Attention_Info(in_dim=args.embedding_dim).to(args.device) for i in
                        range(args.num_heads)]

        # if args.gat_attention_type == 'linear':
        #     self.gat = [LinearAttention(in_dim = gcn_input_dim, mem_dim = gcn_input_dim).to(args.device) for i in range(args.num_heads)] # we prefer to keep the dimension unchanged
        # elif args.gat_attention_type == 'dotprod':
        #     self.gat = [DotprodAttention().to(args.device) for i in range(args.num_heads)]
        # else:
        #     # reshaped gcn
        #     self.gat = nn.Linear(gcn_input_dim, gcn_input_dim)

        # property_embedding
        # self.property_embed = nn.Embedding(property_num, args.embedding_dim)
        # does not fine-tune during training, freeze=True
        self.property_embed = nn.Embedding.from_pretrained(property_embed, freeze=False)

        # last_hidden_size = args.hidden_size * 4
        #
        # layers = [
        #     nn.Linear(last_hidden_size, args.final_hidden_size), nn.ReLU()]
        # for _ in range(args.num_mlps-1):
        #     layers += [nn.Linear(args.final_hidden_size,
        #                          args.final_hidden_size), nn.ReLU()]
        # self.fcs = nn.Sequential(*layers)
        # self.fc_final = nn.Linear(args.final_hidden_size, args.num_classes)
        self.lin1 = nn.Linear(768 * 2, 2)

    # def forward(self, sentence, aspect, pos_class, dep_tags, text_len, aspect_len, dep_rels, dep_heads, aspect_position, dep_dirs):

    def forward(self,
                args,
                target_relation_id,
                subj_property_ids_list,
                obj_property_ids_list,
                subj_property_value_embeds_matrix_list,
                obj_property_value_embeds_matrix_list,
                label_id_list):
        '''
        Forward takes:
            sentence: sentence_id of size (batch_size, text_length)
            aspect: aspect_id of size (batch_size, aspect_length)
            pos_class: pos_tag_id of size (batch_size, text_length)
            dep_tags: dep_tag_id of size (batch_size, text_length)
            text_len: (batch_size,) length of each sentence
            aspect_len: (batch_size, ) aspect length of each sentence
            dep_rels: (batch_size, text_length) relation
            dep_heads: (batch_size, text_length) which node adjacent to that node
            aspect_position: (batch_size, text_length) mask, with the position of aspect as 1 and others as 0
            dep_dirs: (batch_size, text_length) the directions each node to the aspect
        '''
        # fmask = (torch.zeros_like(sentence) != sentence).float()  # (N，L)
        # dmask = (torch.zeros_like(dep_tags) != dep_tags).float()  # (N ,L)

        logits_list = []
        subj_attention_pro_mean_list = []
        obj_attention_pro_mean_list = []

        for i in range(len(label_id_list)):
            relation_embed = self.property_embed(target_relation_id)
            subj_property_feature = self.property_embed(subj_property_ids_list[i])
            obj_property_feature = self.property_embed(obj_property_ids_list[i])

            subj_value_feature = subj_property_value_embeds_matrix_list[i]
            obj_value_feature = obj_property_value_embeds_matrix_list[i]

            # subj_dep_out = [g(subj_value_feature, subj_property_feature).unsqueeze(1) for g in self.gat_dep]  # (N, 1, D) * num_heads
            # obj_dep_out = [g(obj_value_feature, obj_property_feature).unsqueeze(1) for g in self.gat_dep]  # (N, 1, D) * num_heads

            # ---------- fetch the attention probability -------
            # get subject entity information
            subj_dep_out = []
            subj_attention_pro_list = []
            for g in self.gat_dep:
                out, attention_prob = g(subj_value_feature, subj_property_feature)
                out = out.unsqueeze(1)
                subj_dep_out.append(out)
                subj_attention_pro_list.append(attention_prob)
            # endfor

            # get object entity information
            obj_dep_out = []
            obj_attention_pro_list = []
            for g in self.gat_dep:
                out, attention_prob = g(obj_value_feature, obj_property_feature)
                out = out.unsqueeze(1)
                obj_dep_out.append(out)
                obj_attention_pro_list.append(attention_prob)
            # endfor

            subj_attention_pro = np.stack(subj_attention_pro_list)
            obj_attention_pro = np.stack(obj_attention_pro_list)

            subj_attention_pro_mean = np.mean(subj_attention_pro, axis=0)
            obj_attention_pro_mean = np.mean(obj_attention_pro, axis=0)

            subj_attention_pro_mean_list.append(subj_attention_pro_mean)
            obj_attention_pro_mean_list.append(obj_attention_pro_mean)

            subj_dep_out = torch.cat(subj_dep_out, dim=1)  # (N, H, D)
            subj_dep_out = subj_dep_out.mean(dim=1)  # (768,)

            obj_dep_out = torch.cat(obj_dep_out, dim=1)
            obj_dep_out = obj_dep_out.mean(dim=1)  # (768,)

            cat_out = torch.cat([subj_dep_out, obj_dep_out], dim=0)
            lin1_result = self.lin1(cat_out)
            # logit = torch.relu(lin1_result)
            logits_list.append(lin1_result)
        # endfor

        all_logits = torch.stack(logits_list)

        if label_id_list is not None:
            loss_func = nn.CrossEntropyLoss()
            loss = loss_func(all_logits, label_id_list)
            return all_logits, loss, subj_attention_pro_mean_list, obj_attention_pro_mean_list
        else:
            return all_logits, None, subj_attention_pro_mean_list, obj_attention_pro_mean_list

        # aspect_feature = self.embed(aspect)  # (N, L', D)
        #
        # feature = self.dropout(feature)
        # aspect_feature = self.dropout(aspect_feature)
        #
        # if self.args.highway:
        #     feature = self.highway(feature)
        #     aspect_feature = self.highway(aspect_feature)
        #
        # feature, _ = self.bilstm(feature)  # (N,L,D)
        # aspect_feature, _ = self.bilstm(aspect_feature)  # (N,L,D)
        #
        # aspect_feature = aspect_feature.mean(dim=1)  # (N, D)
        #
        # ############################################################################################
        # # do gat thing
        # dep_feature = self.dep_embed(dep_tags)
        # if self.args.highway:
        #     dep_feature = self.highway_dep(dep_feature)

        # dep_out = [g(feature, dep_feature, fmask).unsqueeze(1) for g in self.gat_dep]  # (N, 1, D) * num_heads
        # dep_out = torch.cat(dep_out, dim=1)  # (N, H, D)
        # dep_out = dep_out.mean(dim=1)  # (N, D)
        #
        # if self.args.gat_attention_type == 'gcn':
        #     gat_out = self.gat(feature)  # (N, L, D)
        #     fmask = fmask.unsqueeze(2)
        #     gat_out = gat_out * fmask
        #     gat_out = F.relu(torch.sum(gat_out, dim=1))  # (N, D)
        #
        # else:
        #     gat_out = [g(feature, aspect_feature, fmask).unsqueeze(1) for g in self.gat]
        #     gat_out = torch.cat(gat_out, dim=1)
        #     gat_out = gat_out.mean(dim=1)
        #
        # feature_out = torch.cat([dep_out, gat_out], dim=1)  # (N, D')
        # # feature_out = gat_out
        # #############################################################################################
        # x = self.dropout(feature_out)
        # x = self.fcs(x)
        # logit = self.fc_final(x)
        # return logit


class RGAT_MaxMargin(nn.Module):
    """
    @Deprecated
    relation-aware GAT
    """

    def __init__(self, args, property_embed_tensor, num_of_rc_relation):
        """
        property_num: the total number of properties involved in one-hop of all our entities
        """
        super(RGAT_MaxMargin, self).__init__()
        self.args = args

        args.embedding_dim = 768
        # num_embeddings, embed_dim = args.glove_embedding.shape
        # self.embed = nn.Embedding(num_embeddings, embed_dim)
        # self.embed.weight = nn.Parameter(
        #     args.glove_embedding, requires_grad=False)

        self.dropout = nn.Dropout(args.dropout)
        self.tanh = nn.Tanh()

        # if args.highway:
        #     self.highway_dep = Highway(args.num_layers, args.embedding_dim)
        #     self.highway = Highway(args.num_layers, args.embedding_dim)

        # self.bilstm = nn.LSTM(input_size=args.embedding_dim, hidden_size=args.hidden_size,
        #                       bidirectional=True, batch_first=True, num_layers=args.num_layers)
        # gcn_input_dim = args.hidden_size * 2

        # if args.gat:
        args.num_heads = 8
        self.gat_dep = [
            RelationAttention(num_of_rc_relation=num_of_rc_relation, in_dim=args.embedding_dim).to(args.device) for i in
            range(args.num_heads)]

        # if args.gat_attention_type == 'linear':
        #     self.gat = [LinearAttention(in_dim = gcn_input_dim, mem_dim = gcn_input_dim).to(args.device) for i in range(args.num_heads)] # we prefer to keep the dimension unchanged
        # elif args.gat_attention_type == 'dotprod':
        #     self.gat = [DotprodAttention().to(args.device) for i in range(args.num_heads)]
        # else:
        #     # reshaped gcn
        #     self.gat = nn.Linear(gcn_input_dim, gcn_input_dim)

        # property_embedding
        # self.property_embed = nn.Embedding(property_num, args.embedding_dim)
        # does not fine-tune during training, freeze=True

        self.property_embed = nn.Embedding.from_pretrained(property_embed_tensor, freeze=False)

        # tmp_embed = nn.Embedding(property_embed.size()[0], property_embed.size()[1])
        # nn.init.uniform_(tmp_embed.weight, -1.0, 1.0)
        # self.property_embed = tmp_embed

        # model.embedding.weight.grad
        # torch.manual_seed(3)
        # emb2 = nn.Embedding(5, 5)
        # nn.init.uniform_(emb2.weight, -1.0, 1.0)

        # last_hidden_size = args.hidden_size * 4
        #
        # layers = [
        #     nn.Linear(last_hidden_size, args.final_hidden_size), nn.ReLU()]
        # for _ in range(args.num_mlps-1):
        #     layers += [nn.Linear(args.final_hidden_size,
        #                          args.final_hidden_size), nn.ReLU()]
        # self.fcs = nn.Sequential(*layers)
        # self.fc_final = nn.Linear(args.final_hidden_size, args.num_classes)
        self.lin1 = nn.Linear(768 * 2, 1)
        glorot(self.lin1.weight)
        pass

    def forward(self,
                args,
                head_property_id_tensor_list,
                tail_property_id_tensor_list,
                head_property_value_embeds_matrix_tensor_list,
                tail_property_value_embeds_matrix_tensor_list,
                nd_label_id_list,
                rc_label_id_list):
        '''
        Forward takes:
            sentence: sentence_id of size (batch_size, text_length)
            aspect: aspect_id of size (batch_size, aspect_length)
            pos_class: pos_tag_id of size (batch_size, text_length)
            dep_tags: dep_tag_id of size (batch_size, text_length)
            text_len: (batch_size,) length of each sentence
            aspect_len: (batch_size, ) aspect length of each sentence
            dep_rels: (batch_size, text_length) relation
            dep_heads: (batch_size, text_length) which node adjacent to that node
            aspect_position: (batch_size, text_length) mask, with the position of aspect as 1 and others as 0
            dep_dirs: (batch_size, text_length) the directions each node to the aspect
        '''
        # fmask = (torch.zeros_like(sentence) != sentence).float()  # (N，L)
        # dmask = (torch.zeros_like(dep_tags) != dep_tags).float()  # (N ,L)

        batch_size = len(head_property_id_tensor_list)

        cat_out_list = []
        for i in range(len(rc_label_id_list)):
            # relation_embed = self.property_embed(target_relation_id)
            rc_label_id = rc_label_id_list[i]

            head_property_feature = self.property_embed(head_property_id_tensor_list[i])
            tail_property_feature = self.property_embed(tail_property_id_tensor_list[i])

            head_value_feature = head_property_value_embeds_matrix_tensor_list[i]
            tail_value_feature = tail_property_value_embeds_matrix_tensor_list[i]

            # here, it uses [0], to only use the first return value
            # (N, 1, D) * num_heads
            head_property_out = [g(head_value_feature, head_property_feature, rc_label_id)[0].unsqueeze(1) for g in
                                 self.gat_dep]
            # (N, 1, D) * num_heads
            tail_property_out = [g(tail_value_feature, tail_property_feature, rc_label_id)[0].unsqueeze(1) for g in
                                 self.gat_dep]

            head_property_out = torch.cat(head_property_out, dim=1)  # (N, H, D)
            head_property_out = head_property_out.mean(dim=1)  # (768,)

            tail_property_out = torch.cat(tail_property_out, dim=1)
            tail_property_out = tail_property_out.mean(dim=1)  # (768,)

            cat_out = torch.cat([head_property_out, tail_property_out], dim=0)

            cat_out = torch.unsqueeze(cat_out, dim=0)

            cat_out_list.append(cat_out)
        # endfor

        target_embed_list = torch.cat(cat_out_list, dim=0)
        logits = self.lin1(target_embed_list)
        total_score = torch.squeeze(logits, dim=1)

        loss = None
        if nd_label_id_list is not None:
            # Note: get positive / negative logits here
            # positive
            pos_mask = nd_label_id_list.eq(1)
            pos_score = total_score[pos_mask]
            # negative
            neg_mask = nd_label_id_list.eq(0)
            neg_score = total_score[neg_mask]

            # Note: max margin loss
            max_margin_loss = nn.MarginRankingLoss(margin=1)
            target = torch.tensor([1] * int(batch_size / 2), dtype=torch.long).to(args.device)
            loss = max_margin_loss(pos_score, neg_score, target)
        # endif

        return total_score, loss


class RGAT_Stack(nn.Module):
    """
    relation-aware GAT

    The last linear transform layer is also stacked
    """

    def __init__(self, args, property_embed_tensor, num_of_rc_relation):
        """
        property_num: the total number of properties involved in one-hop of all our entities
        """
        super(RGAT_Stack, self).__init__()
        self.args = args

        args.embedding_dim = 768
        # num_embeddings, embed_dim = args.glove_embedding.shape
        # self.embed = nn.Embedding(num_embeddings, embed_dim)
        # self.embed.weight = nn.Parameter(
        #     args.glove_embedding, requires_grad=False)

        self.dropout = nn.Dropout(args.dropout)
        self.tanh = nn.Tanh()

        # if args.highway:
        #     self.highway_dep = Highway(args.num_layers, args.embedding_dim)
        #     self.highway = Highway(args.num_layers, args.embedding_dim)

        # self.bilstm = nn.LSTM(input_size=args.embedding_dim, hidden_size=args.hidden_size,
        #                       bidirectional=True, batch_first=True, num_layers=args.num_layers)
        # gcn_input_dim = args.hidden_size * 2

        # if args.gat:
        args.num_heads = 8
        self.gat_dep = [
            RelationAttention(num_of_rc_relation=num_of_rc_relation, in_dim=args.embedding_dim).to(args.device) for i in
            range(args.num_heads)]

        # if args.gat_attention_type == 'linear':
        #     self.gat = [LinearAttention(in_dim = gcn_input_dim, mem_dim = gcn_input_dim).to(args.device) for i in range(args.num_heads)] # we prefer to keep the dimension unchanged
        # elif args.gat_attention_type == 'dotprod':
        #     self.gat = [DotprodAttention().to(args.device) for i in range(args.num_heads)]
        # else:
        #     # reshaped gcn
        #     self.gat = nn.Linear(gcn_input_dim, gcn_input_dim)

        # property_embedding
        # self.property_embed = nn.Embedding(property_num, args.embedding_dim)
        # does not fine-tune during training, freeze=True

        self.property_embed = nn.Embedding.from_pretrained(property_embed_tensor, freeze=False)

        # tmp_embed = nn.Embedding(property_embed.size()[0], property_embed.size()[1])
        # nn.init.uniform_(tmp_embed.weight, -1.0, 1.0)
        # self.property_embed = tmp_embed

        # model.embedding.weight.grad
        # torch.manual_seed(3)
        # emb2 = nn.Embedding(5, 5)
        # nn.init.uniform_(emb2.weight, -1.0, 1.0)

        # last_hidden_size = args.hidden_size * 4
        #
        # layers = [
        #     nn.Linear(last_hidden_size, args.final_hidden_size), nn.ReLU()]
        # for _ in range(args.num_mlps-1):
        #     layers += [nn.Linear(args.final_hidden_size,
        #                          args.final_hidden_size), nn.ReLU()]
        # self.fcs = nn.Sequential(*layers)
        # self.fc_final = nn.Linear(args.final_hidden_size, args.num_classes)

        # self.lin1 = nn.Linear(768 * 2, 1)
        # glorot(self.lin1.weight)

        # last layer transform
        transform_linear_list = [nn.Linear(768 * 2, 2) for i in range(num_of_rc_relation)]

        # initialize
        for layer in transform_linear_list:
            glorot(layer.weight)
        # endfor

        self.lin1_list = torch.nn.ModuleList(transform_linear_list)

        pass

    def forward(self,
                args,
                head_property_id_tensor_list,
                tail_property_id_tensor_list,
                head_property_value_embeds_matrix_tensor_list,
                tail_property_value_embeds_matrix_tensor_list,
                nd_label_id_list,
                rc_label_id_list):
        '''
        Forward takes:
            sentence: sentence_id of size (batch_size, text_length)
            aspect: aspect_id of size (batch_size, aspect_length)
            pos_class: pos_tag_id of size (batch_size, text_length)
            dep_tags: dep_tag_id of size (batch_size, text_length)
            text_len: (batch_size,) length of each sentence
            aspect_len: (batch_size, ) aspect length of each sentence
            dep_rels: (batch_size, text_length) relation
            dep_heads: (batch_size, text_length) which node adjacent to that node
            aspect_position: (batch_size, text_length) mask, with the position of aspect as 1 and others as 0
            dep_dirs: (batch_size, text_length) the directions each node to the aspect
        '''
        # fmask = (torch.zeros_like(sentence) != sentence).float()  # (N，L)
        # dmask = (torch.zeros_like(dep_tags) != dep_tags).float()  # (N ,L)

        batch_size = len(head_property_id_tensor_list)

        total_score = []
        all_logits = []
        for i in range(len(rc_label_id_list)):
            # relation_embed = self.property_embed(target_relation_id)
            rc_label_id = rc_label_id_list[i]

            head_property_feature = self.property_embed(head_property_id_tensor_list[i])
            tail_property_feature = self.property_embed(tail_property_id_tensor_list[i])

            head_value_feature = head_property_value_embeds_matrix_tensor_list[i]
            tail_value_feature = tail_property_value_embeds_matrix_tensor_list[i]

            # here, it uses [0], to only use the first return value
            # (N, 1, D) * num_heads
            head_property_out = [g(head_value_feature, head_property_feature, rc_label_id)[0].unsqueeze(1) for g in
                                 self.gat_dep]
            # (N, 1, D) * num_heads
            tail_property_out = [g(tail_value_feature, tail_property_feature, rc_label_id)[0].unsqueeze(1) for g in
                                 self.gat_dep]

            head_property_out = torch.cat(head_property_out, dim=1)  # (N, H, D)
            head_property_out = head_property_out.mean(dim=1)  # (768,)

            tail_property_out = torch.cat(tail_property_out, dim=1)
            tail_property_out = tail_property_out.mean(dim=1)  # (768,)

            # each is transformed by its own linear layer
            cat_out = torch.cat([head_property_out, tail_property_out], dim=0)

            cat_out = torch.unsqueeze(cat_out, dim=0)  # (1, 1536)

            score = self.lin1_list[rc_label_id](cat_out)  # (1, 1536) * (1536, 1) -> (1, 1)
            score = torch.squeeze(score, dim=0)
            all_logits.append(score)

        # endfor

        # total_score = torch.cat(total_score, dim=1)
        # total_score = torch.squeeze(total_score, dim=0)

        all_logits = torch.stack(all_logits)

        loss = None
        if nd_label_id_list is not None:
            loss_func = nn.CrossEntropyLoss()
            loss = loss_func(all_logits, nd_label_id_list)
        # endif

        return all_logits, loss

        # loss = None
        # if nd_label_id_list is not None:
        #     # Note: get positive / negative logits here
        #     # positive
        #     pos_mask = nd_label_id_list.eq(1)
        #     pos_score = total_score[pos_mask]
        #     # negative
        #     neg_mask = nd_label_id_list.eq(0)
        #     neg_score = total_score[neg_mask]
        #
        #     # Note: max margin loss
        #     max_margin_loss = nn.MarginRankingLoss(margin=1)
        #     target = torch.tensor([1] * int(batch_size / 2), dtype=torch.long).to(args.device)
        #     loss = max_margin_loss(pos_score, neg_score, target)
        # # endif

        # cat_out = torch.cat([head_dep_out, tail_dep_out], dim=0)
        #
        #             lin1_result = self.lin1(cat_out)
        #             # logit = torch.relu(lin1_result)
        #             logits_list.append(lin1_result)
        #         # endfor
        #
        #         all_logits = torch.stack(logits_list)
        #
        #         if nd_label_id_list is not None:
        #             loss_func = nn.CrossEntropyLoss()
        #             loss = loss_func(all_logits, nd_label_id_list)
        #             return all_logits, loss
        #         else:
        #             return all_logits, None


class RGAT_MaxMargin_Stack(nn.Module):
    """
    relation-aware GAT

    The last linear transform layer is also stacked
    """

    def __init__(self, args, property_embed_tensor, num_of_rc_relation):
        """
        property_num: the total number of properties involved in one-hop of all our entities
        """
        super(RGAT_MaxMargin_Stack, self).__init__()
        self.args = args

        args.embedding_dim = 768
        # num_embeddings, embed_dim = args.glove_embedding.shape
        # self.embed = nn.Embedding(num_embeddings, embed_dim)
        # self.embed.weight = nn.Parameter(
        #     args.glove_embedding, requires_grad=False)

        self.dropout = nn.Dropout(args.dropout)
        self.tanh = nn.Tanh()

        # if args.highway:
        #     self.highway_dep = Highway(args.num_layers, args.embedding_dim)
        #     self.highway = Highway(args.num_layers, args.embedding_dim)

        # self.bilstm = nn.LSTM(input_size=args.embedding_dim, hidden_size=args.hidden_size,
        #                       bidirectional=True, batch_first=True, num_layers=args.num_layers)
        # gcn_input_dim = args.hidden_size * 2

        # if args.gat:
        args.num_heads = 8
        self.gat_dep = [
            RelationAttention(num_of_rc_relation=num_of_rc_relation, in_dim=args.embedding_dim).to(args.device) for i in
            range(args.num_heads)]

        # if args.gat_attention_type == 'linear':
        #     self.gat = [LinearAttention(in_dim = gcn_input_dim, mem_dim = gcn_input_dim).to(args.device) for i in range(args.num_heads)] # we prefer to keep the dimension unchanged
        # elif args.gat_attention_type == 'dotprod':
        #     self.gat = [DotprodAttention().to(args.device) for i in range(args.num_heads)]
        # else:
        #     # reshaped gcn
        #     self.gat = nn.Linear(gcn_input_dim, gcn_input_dim)

        # property_embedding
        # self.property_embed = nn.Embedding(property_num, args.embedding_dim)
        # does not fine-tune during training, freeze=True

        self.property_embed = nn.Embedding.from_pretrained(property_embed_tensor, freeze=False)

        # tmp_embed = nn.Embedding(property_embed.size()[0], property_embed.size()[1])
        # nn.init.uniform_(tmp_embed.weight, -1.0, 1.0)
        # self.property_embed = tmp_embed

        # model.embedding.weight.grad
        # torch.manual_seed(3)
        # emb2 = nn.Embedding(5, 5)
        # nn.init.uniform_(emb2.weight, -1.0, 1.0)

        # last_hidden_size = args.hidden_size * 4
        #
        # layers = [
        #     nn.Linear(last_hidden_size, args.final_hidden_size), nn.ReLU()]
        # for _ in range(args.num_mlps-1):
        #     layers += [nn.Linear(args.final_hidden_size,
        #                          args.final_hidden_size), nn.ReLU()]
        # self.fcs = nn.Sequential(*layers)
        # self.fc_final = nn.Linear(args.final_hidden_size, args.num_classes)

        # self.lin1 = nn.Linear(768 * 2, 1)
        # glorot(self.lin1.weight)

        # last layer transform
        transform_linear_list = [nn.Linear(768 * 2, 1) for i in range(num_of_rc_relation)]

        # initialize
        for layer in transform_linear_list:
            glorot(layer.weight)
        # endfor

        self.lin1_list = torch.nn.ModuleList(transform_linear_list)

        pass

    def forward(self,
                args,
                head_property_id_tensor_list,
                tail_property_id_tensor_list,
                head_property_value_embeds_matrix_tensor_list,
                tail_property_value_embeds_matrix_tensor_list,
                nd_label_id_list,
                rc_label_id_list):
        '''
        Forward takes:
            sentence: sentence_id of size (batch_size, text_length)
            aspect: aspect_id of size (batch_size, aspect_length)
            pos_class: pos_tag_id of size (batch_size, text_length)
            dep_tags: dep_tag_id of size (batch_size, text_length)
            text_len: (batch_size,) length of each sentence
            aspect_len: (batch_size, ) aspect length of each sentence
            dep_rels: (batch_size, text_length) relation
            dep_heads: (batch_size, text_length) which node adjacent to that node
            aspect_position: (batch_size, text_length) mask, with the position of aspect as 1 and others as 0
            dep_dirs: (batch_size, text_length) the directions each node to the aspect
        '''
        # fmask = (torch.zeros_like(sentence) != sentence).float()  # (N，L)
        # dmask = (torch.zeros_like(dep_tags) != dep_tags).float()  # (N ,L)

        batch_size = len(head_property_id_tensor_list)

        total_score = []
        for i in range(len(rc_label_id_list)):
            # relation_embed = self.property_embed(target_relation_id)
            rc_label_id = rc_label_id_list[i]

            head_property_feature = self.property_embed(head_property_id_tensor_list[i])
            tail_property_feature = self.property_embed(tail_property_id_tensor_list[i])

            head_value_feature = head_property_value_embeds_matrix_tensor_list[i]
            tail_value_feature = tail_property_value_embeds_matrix_tensor_list[i]

            # here, it uses [0], to only use the first return value
            # (N, 1, D) * num_heads
            head_property_out = [g(head_value_feature, head_property_feature, rc_label_id)[0].unsqueeze(1) for g in
                                 self.gat_dep]
            # (N, 1, D) * num_heads
            tail_property_out = [g(tail_value_feature, tail_property_feature, rc_label_id)[0].unsqueeze(1) for g in
                                 self.gat_dep]

            head_property_out = torch.cat(head_property_out, dim=1)  # (N, H, D)
            head_property_out = head_property_out.mean(dim=1)  # (768,)

            tail_property_out = torch.cat(tail_property_out, dim=1)
            tail_property_out = tail_property_out.mean(dim=1)  # (768,)

            # each is transformed by its own linear layer
            cat_out = torch.cat([head_property_out, tail_property_out], dim=0)

            cat_out = torch.unsqueeze(cat_out, dim=0)  # (1, 1536)

            score = self.lin1_list[rc_label_id](cat_out)  # (1, 1536) * (1536, 1) -> (1, 1)

            total_score.append(score)
        # endfor

        total_score = torch.cat(total_score, dim=1)
        total_score = torch.squeeze(total_score, dim=0)

        loss = None
        if nd_label_id_list is not None:
            # Note: get positive / negative logits here
            # positive
            pos_mask = nd_label_id_list.eq(1)
            pos_score = total_score[pos_mask]
            # negative
            neg_mask = nd_label_id_list.eq(0)
            neg_score = total_score[neg_mask]

            # Note: max margin loss
            max_margin_loss = nn.MarginRankingLoss(margin=1)
            target = torch.tensor([1] * int(batch_size / 2), dtype=torch.long).to(args.device)
            loss = max_margin_loss(pos_score, neg_score, target)
        # endif

        return total_score, loss


class RGAT_MaxMargin_Stack_with_Attention_Info(nn.Module):
    """
    relation-aware GAT

    The last linear transform layer is also stacked
    """

    def __init__(self, args, property_embed_tensor, num_of_rc_relation):
        """
        property_num: the total number of properties involved in one-hop of all our entities
        """
        super(RGAT_MaxMargin_Stack_with_Attention_Info, self).__init__()
        self.args = args

        args.embedding_dim = 768
        # num_embeddings, embed_dim = args.glove_embedding.shape
        # self.embed = nn.Embedding(num_embeddings, embed_dim)
        # self.embed.weight = nn.Parameter(
        #     args.glove_embedding, requires_grad=False)

        self.dropout = nn.Dropout(args.dropout)
        self.tanh = nn.Tanh()

        # if args.highway:
        #     self.highway_dep = Highway(args.num_layers, args.embedding_dim)
        #     self.highway = Highway(args.num_layers, args.embedding_dim)

        # self.bilstm = nn.LSTM(input_size=args.embedding_dim, hidden_size=args.hidden_size,
        #                       bidirectional=True, batch_first=True, num_layers=args.num_layers)
        # gcn_input_dim = args.hidden_size * 2

        # if args.gat:
        args.num_heads = 8
        self.gat_dep = [
            RelationAttention(num_of_rc_relation=num_of_rc_relation, in_dim=args.embedding_dim).to(args.device) for i in
            range(args.num_heads)]

        # if args.gat_attention_type == 'linear':
        #     self.gat = [LinearAttention(in_dim = gcn_input_dim, mem_dim = gcn_input_dim).to(args.device) for i in range(args.num_heads)] # we prefer to keep the dimension unchanged
        # elif args.gat_attention_type == 'dotprod':
        #     self.gat = [DotprodAttention().to(args.device) for i in range(args.num_heads)]
        # else:
        #     # reshaped gcn
        #     self.gat = nn.Linear(gcn_input_dim, gcn_input_dim)

        # property_embedding
        # self.property_embed = nn.Embedding(property_num, args.embedding_dim)
        # does not fine-tune during training, freeze=True

        self.property_embed = nn.Embedding.from_pretrained(property_embed_tensor, freeze=False)

        # tmp_embed = nn.Embedding(property_embed.size()[0], property_embed.size()[1])
        # nn.init.uniform_(tmp_embed.weight, -1.0, 1.0)
        # self.property_embed = tmp_embed

        # model.embedding.weight.grad
        # torch.manual_seed(3)
        # emb2 = nn.Embedding(5, 5)
        # nn.init.uniform_(emb2.weight, -1.0, 1.0)

        # last_hidden_size = args.hidden_size * 4
        #
        # layers = [
        #     nn.Linear(last_hidden_size, args.final_hidden_size), nn.ReLU()]
        # for _ in range(args.num_mlps-1):
        #     layers += [nn.Linear(args.final_hidden_size,
        #                          args.final_hidden_size), nn.ReLU()]
        # self.fcs = nn.Sequential(*layers)
        # self.fc_final = nn.Linear(args.final_hidden_size, args.num_classes)

        # self.lin1 = nn.Linear(768 * 2, 1)
        # glorot(self.lin1.weight)

        # last layer transform
        transform_linear_list = [nn.Linear(768 * 2, 1) for i in range(num_of_rc_relation)]

        # initialize
        for layer in transform_linear_list:
            glorot(layer.weight)
        # endfor

        self.lin1_list = torch.nn.ModuleList(transform_linear_list)

        pass

    def forward(self,
                args,
                cur_batch_data_guid_list,
                head_property_id_tensor_list,
                tail_property_id_tensor_list,
                head_property_value_embeds_matrix_tensor_list,
                tail_property_value_embeds_matrix_tensor_list,
                nd_label_id_list,
                rc_label_id_list):
        '''
        Forward takes:
            sentence: sentence_id of size (batch_size, text_length)
            aspect: aspect_id of size (batch_size, aspect_length)
            pos_class: pos_tag_id of size (batch_size, text_length)
            dep_tags: dep_tag_id of size (batch_size, text_length)
            text_len: (batch_size,) length of each sentence
            aspect_len: (batch_size, ) aspect length of each sentence
            dep_rels: (batch_size, text_length) relation
            dep_heads: (batch_size, text_length) which node adjacent to that node
            aspect_position: (batch_size, text_length) mask, with the position of aspect as 1 and others as 0
            dep_dirs: (batch_size, text_length) the directions each node to the aspect
        '''
        # fmask = (torch.zeros_like(sentence) != sentence).float()  # (N，L)
        # dmask = (torch.zeros_like(dep_tags) != dep_tags).float()  # (N ,L)

        batch_size = len(head_property_id_tensor_list)

        total_score = []

        # ######## attention information #########
        head_attention_prob_mean_list = []
        tail_attention_prob_mean_list = []

        for i in range(len(rc_label_id_list)):
            
            if cur_batch_data_guid_list[i] == "P161-4-test-s-1a8e70c1-6bed-421e-a696-adf8a6a7687e":
                print("hello")
            
            # relation_embed = self.property_embed(target_relation_id)
            rc_label_id = rc_label_id_list[i]

            head_property_feature = self.property_embed(head_property_id_tensor_list[i])
            tail_property_feature = self.property_embed(tail_property_id_tensor_list[i])

            head_value_feature = head_property_value_embeds_matrix_tensor_list[i]
            tail_value_feature = tail_property_value_embeds_matrix_tensor_list[i]

            # ######### (1) entity embed from RGAT #######
            # ######### (2) RGAT attention prob #######

            # #### head ####
            head_property_out_list = []
            head_att_prob_list = []
            for g in self.gat_dep:
                head_out, att_prob = g(head_value_feature, head_property_feature, rc_label_id)
                head_out = head_out.unsqueeze(1)
                head_property_out_list.append(head_out)
                head_att_prob_list.append(att_prob)
            # endfor

            # #### tail ####
            tail_property_out_list = []
            tail_att_prob_list = []
            for g in self.gat_dep:
                tail_out, att_prob = g(tail_value_feature, tail_property_feature, rc_label_id)
                tail_out = tail_out.unsqueeze(1)
                tail_property_out_list.append(tail_out)
                tail_att_prob_list.append(att_prob)
            # endfor

            # # here, it uses [0], to only use the first return value
            # # (N, 1, D) * num_heads
            # head_property_out = [g(head_value_feature, head_property_feature, rc_label_id)[0].unsqueeze(1) for g in
            #                      self.gat_dep]
            # # (N, 1, D) * num_heads
            # tail_property_out = [g(tail_value_feature, tail_property_feature, rc_label_id)[0].unsqueeze(1) for g in
            #                      self.gat_dep]

            # #### take mean for entity embed #####
            head_property_out = torch.cat(head_property_out_list, dim=1)  # (N, H, D)
            head_property_out = head_property_out.mean(dim=1)  # (768,)

            tail_property_out = torch.cat(tail_property_out_list, dim=1)
            tail_property_out = tail_property_out.mean(dim=1)  # (768,)

            # ##### take mean for att prob ######
            head_att_prob = np.stack(head_att_prob_list)
            head_att_prob_mean = np.mean(head_att_prob, axis=0)
            head_attention_prob_mean_list.append(head_att_prob_mean)

            tail_att_prob = np.stack(tail_att_prob_list)
            tail_att_prob_mean = np.mean(tail_att_prob, axis=0)
            tail_attention_prob_mean_list.append(tail_att_prob_mean)

            # ######## calculate score ########
            # (1)
            cat_out = torch.cat([head_property_out, tail_property_out], dim=0)
            cat_out = torch.unsqueeze(cat_out, dim=0)  # (1, 1536)
            # (2) each is transformed by its own linear layer
            score = self.lin1_list[rc_label_id](cat_out)  # (1, 1536) * (1536, 1) -> (1, 1)
            total_score.append(score)
            # ###########

        # endfor

        total_score = torch.cat(total_score, dim=1)
        total_score = torch.squeeze(total_score, dim=0)

        loss = None
        if nd_label_id_list is not None:
            # Note: get positive / negative logits here
            # positive
            pos_mask = nd_label_id_list.eq(1)
            pos_score = total_score[pos_mask]
            # negative
            neg_mask = nd_label_id_list.eq(0)
            neg_score = total_score[neg_mask]

            # Note: max margin loss
            max_margin_loss = nn.MarginRankingLoss(margin=1)
            target = torch.tensor([1] * int(batch_size / 2), dtype=torch.long).to(args.device)
            loss = max_margin_loss(pos_score, neg_score, target)
        # endif

        return total_score, loss, head_attention_prob_mean_list, tail_attention_prob_mean_list



class RGAT_Stack_With_Head_Tail_Attention_CrossEntropy(nn.Module):
    """
    relation-aware GAT

    The last linear transform layer is also stacked
    """

    def __init__(self, 
                 args, 
                 property_embed_tensor, 
                 relation_str_to_index_dict,
                 index_to_relation_str_dict):
        """
        property_num: the total number of properties involved in one-hop of all our entities
        """
        super(RGAT_Stack_With_Head_Tail_Attention_CrossEntropy, self).__init__()
        self.args = args

        args.embedding_dim = 768
        # num_embeddings, embed_dim = args.glove_embedding.shape
        # self.embed = nn.Embedding(num_embeddings, embed_dim)
        # self.embed.weight = nn.Parameter(
        #     args.glove_embedding, requires_grad=False)

        self.dropout = nn.Dropout(args.dropout)
        self.tanh = nn.Tanh()

        # if args.highway:
        #     self.highway_dep = Highway(args.num_layers, args.embedding_dim)
        #     self.highway = Highway(args.num_layers, args.embedding_dim)

        # self.bilstm = nn.LSTM(input_size=args.embedding_dim, hidden_size=args.hidden_size,
        #                       bidirectional=True, batch_first=True, num_layers=args.num_layers)
        # gcn_input_dim = args.hidden_size * 2
        num_of_rc_relation = len(relation_str_to_index_dict)

        # if args.gat:
        args.num_heads = 8
        
        # head Attention
        self.head_att = [
            RelationAttention(num_of_rc_relation=num_of_rc_relation, in_dim=args.embedding_dim).to(args.device) for i in
            range(args.num_heads)]
        
        # Tail Attention
        self.tail_att = [
            RelationAttention(num_of_rc_relation=num_of_rc_relation, in_dim=args.embedding_dim).to(args.device) for i in
            range(args.num_heads)]

        # if args.gat_attention_type == 'linear':
        #     self.gat = [LinearAttention(in_dim = gcn_input_dim, mem_dim = gcn_input_dim).to(args.device) for i in range(args.num_heads)] # we prefer to keep the dimension unchanged
        # elif args.gat_attention_type == 'dotprod':
        #     self.gat = [DotprodAttention().to(args.device) for i in range(args.num_heads)]
        # else:
        #     # reshaped gcn
        #     self.gat = nn.Linear(gcn_input_dim, gcn_input_dim)

        # property_embedding
        # self.property_embed = nn.Embedding(property_num, args.embedding_dim)
        # does not fine-tune during training, freeze=True

        self.property_embed = nn.Embedding.from_pretrained(property_embed_tensor, freeze=False)

        # tmp_embed = nn.Embedding(property_embed.size()[0], property_embed.size()[1])
        # nn.init.uniform_(tmp_embed.weight, -1.0, 1.0)
        # self.property_embed = tmp_embed

        # model.embedding.weight.grad
        # torch.manual_seed(3)
        # emb2 = nn.Embedding(5, 5)
        # nn.init.uniform_(emb2.weight, -1.0, 1.0)

        # last_hidden_size = args.hidden_size * 4
        #
        # layers = [
        #     nn.Linear(last_hidden_size, args.final_hidden_size), nn.ReLU()]
        # for _ in range(args.num_mlps-1):
        #     layers += [nn.Linear(args.final_hidden_size,
        #                          args.final_hidden_size), nn.ReLU()]
        # self.fcs = nn.Sequential(*layers)
        # self.fc_final = nn.Linear(args.final_hidden_size, args.num_classes)

        # self.lin1 = nn.Linear(768 * 2, 1)
        # glorot(self.lin1.weight)

        # last layer transform
        transform_linear_list = [nn.Linear(768 * 2, 2) for i in range(num_of_rc_relation)]

        # initialize
        for layer in transform_linear_list:
            glorot(layer.weight)
        # endfor

        self.lin1_list = torch.nn.ModuleList(transform_linear_list)

        pass

    def forward(self,
                args,
                head_property_id_tensor_list,
                tail_property_id_tensor_list,
                head_property_value_embeds_matrix_tensor_list,
                tail_property_value_embeds_matrix_tensor_list,
                nd_label_id_list,
                rc_label_id_list):
        '''
        Forward takes:
            sentence: sentence_id of size (batch_size, text_length)
            aspect: aspect_id of size (batch_size, aspect_length)
            pos_class: pos_tag_id of size (batch_size, text_length)
            dep_tags: dep_tag_id of size (batch_size, text_length)
            text_len: (batch_size,) length of each sentence
            aspect_len: (batch_size, ) aspect length of each sentence
            dep_rels: (batch_size, text_length) relation
            dep_heads: (batch_size, text_length) which node adjacent to that node
            aspect_position: (batch_size, text_length) mask, with the position of aspect as 1 and others as 0
            dep_dirs: (batch_size, text_length) the directions each node to the aspect
        '''
        # fmask = (torch.zeros_like(sentence) != sentence).float()  # (N，L)
        # dmask = (torch.zeros_like(dep_tags) != dep_tags).float()  # (N ,L)

        batch_size = len(head_property_id_tensor_list)

        # total_score = []
        logits_list = []
        for i in range(len(rc_label_id_list)):
            # relation_embed = self.property_embed(target_relation_id)
            rc_label_id = rc_label_id_list[i]

            head_property_feature = self.property_embed(head_property_id_tensor_list[i])
            tail_property_feature = self.property_embed(tail_property_id_tensor_list[i])

            head_value_feature = head_property_value_embeds_matrix_tensor_list[i]
            tail_value_feature = tail_property_value_embeds_matrix_tensor_list[i]

            # here, it uses [0], to only use the first return value
            # (N, 1, D) * num_heads
            head_property_out = [g(head_value_feature, head_property_feature, rc_label_id)[0].unsqueeze(1) for g in
                                 self.head_att]
            # (N, 1, D) * num_heads
            tail_property_out = [g(tail_value_feature, tail_property_feature, rc_label_id)[0].unsqueeze(1) for g in
                                 self.tail_att]

            head_property_out = torch.cat(head_property_out, dim=1)  # (N, H, D)
            head_property_out = head_property_out.mean(dim=1)  # (768,)

            tail_property_out = torch.cat(tail_property_out, dim=1)
            tail_property_out = tail_property_out.mean(dim=1)  # (768,)

            # each is transformed by its own linear layer
            cat_out = torch.cat([head_property_out, tail_property_out], dim=0)

            cat_out = torch.unsqueeze(cat_out, dim=0)  # (1, 1536)

            lin1_result = self.lin1_list[rc_label_id](cat_out)  # (1, 1536) * (1536, 2) -> (1, 2)

            logits_list.append(lin1_result)
            # total_score.append(score)
        # endfor
        
        all_logits = torch.cat(logits_list, dim=0)  # (batch_size, 2)
        
        all_scores = torch.argmax(all_logits, dim=1)
        
        if nd_label_id_list is not None:
            loss_func = nn.CrossEntropyLoss()
            loss = loss_func(all_logits, nd_label_id_list)
            return all_scores, loss
        else:
            return all_scores, None


class RGAT_MaxMargin_Stack_With_Head_Tail_Attention(nn.Module):
    """
    relation-aware GAT

    The last linear transform layer is also stacked
    """

    def __init__(self, 
                 args, 
                 property_embed_tensor, 
                 relation_str_to_index_dict,
                 index_to_relation_str_dict):
        """
        property_num: the total number of properties involved in one-hop of all our entities
        """
        super(RGAT_MaxMargin_Stack_With_Head_Tail_Attention, self).__init__()
        self.args = args

        args.embedding_dim = 768
        # num_embeddings, embed_dim = args.glove_embedding.shape
        # self.embed = nn.Embedding(num_embeddings, embed_dim)
        # self.embed.weight = nn.Parameter(
        #     args.glove_embedding, requires_grad=False)

        self.dropout = nn.Dropout(args.dropout)
        self.tanh = nn.Tanh()

        # if args.highway:
        #     self.highway_dep = Highway(args.num_layers, args.embedding_dim)
        #     self.highway = Highway(args.num_layers, args.embedding_dim)

        # self.bilstm = nn.LSTM(input_size=args.embedding_dim, hidden_size=args.hidden_size,
        #                       bidirectional=True, batch_first=True, num_layers=args.num_layers)
        # gcn_input_dim = args.hidden_size * 2
        num_of_rc_relation = len(relation_str_to_index_dict)

        # if args.gat:
        args.num_heads = 8
        
        # head Attention
        self.head_att = [
            RelationAttention(num_of_rc_relation=num_of_rc_relation, in_dim=args.embedding_dim).to(args.device) for i in
            range(args.num_heads)]
        
        # Tail Attention
        self.tail_att = [
            RelationAttention(num_of_rc_relation=num_of_rc_relation, in_dim=args.embedding_dim).to(args.device) for i in
            range(args.num_heads)]

        # if args.gat_attention_type == 'linear':
        #     self.gat = [LinearAttention(in_dim = gcn_input_dim, mem_dim = gcn_input_dim).to(args.device) for i in range(args.num_heads)] # we prefer to keep the dimension unchanged
        # elif args.gat_attention_type == 'dotprod':
        #     self.gat = [DotprodAttention().to(args.device) for i in range(args.num_heads)]
        # else:
        #     # reshaped gcn
        #     self.gat = nn.Linear(gcn_input_dim, gcn_input_dim)

        # property_embedding
        # self.property_embed = nn.Embedding(property_num, args.embedding_dim)
        # does not fine-tune during training, freeze=True

        self.property_embed = nn.Embedding.from_pretrained(property_embed_tensor, freeze=False)

        # tmp_embed = nn.Embedding(property_embed.size()[0], property_embed.size()[1])
        # nn.init.uniform_(tmp_embed.weight, -1.0, 1.0)
        # self.property_embed = tmp_embed

        # model.embedding.weight.grad
        # torch.manual_seed(3)
        # emb2 = nn.Embedding(5, 5)
        # nn.init.uniform_(emb2.weight, -1.0, 1.0)

        # last_hidden_size = args.hidden_size * 4
        #
        # layers = [
        #     nn.Linear(last_hidden_size, args.final_hidden_size), nn.ReLU()]
        # for _ in range(args.num_mlps-1):
        #     layers += [nn.Linear(args.final_hidden_size,
        #                          args.final_hidden_size), nn.ReLU()]
        # self.fcs = nn.Sequential(*layers)
        # self.fc_final = nn.Linear(args.final_hidden_size, args.num_classes)

        # self.lin1 = nn.Linear(768 * 2, 1)
        # glorot(self.lin1.weight)

        # last layer transform
        transform_linear_list = [nn.Linear(768 * 2, 1) for i in range(num_of_rc_relation)]

        # initialize
        for layer in transform_linear_list:
            glorot(layer.weight)
        # endfor

        self.lin1_list = torch.nn.ModuleList(transform_linear_list)

        pass

    def forward(self,
                args,
                head_property_id_tensor_list,
                tail_property_id_tensor_list,
                head_property_value_embeds_matrix_tensor_list,
                tail_property_value_embeds_matrix_tensor_list,
                nd_label_id_list,
                rc_label_id_list):
        '''
        Forward takes:
            sentence: sentence_id of size (batch_size, text_length)
            aspect: aspect_id of size (batch_size, aspect_length)
            pos_class: pos_tag_id of size (batch_size, text_length)
            dep_tags: dep_tag_id of size (batch_size, text_length)
            text_len: (batch_size,) length of each sentence
            aspect_len: (batch_size, ) aspect length of each sentence
            dep_rels: (batch_size, text_length) relation
            dep_heads: (batch_size, text_length) which node adjacent to that node
            aspect_position: (batch_size, text_length) mask, with the position of aspect as 1 and others as 0
            dep_dirs: (batch_size, text_length) the directions each node to the aspect
        '''
        # fmask = (torch.zeros_like(sentence) != sentence).float()  # (N，L)
        # dmask = (torch.zeros_like(dep_tags) != dep_tags).float()  # (N ,L)

        batch_size = len(head_property_id_tensor_list)

        total_score = []
        for i in range(len(rc_label_id_list)):
            # relation_embed = self.property_embed(target_relation_id)
            rc_label_id = rc_label_id_list[i]

            head_property_feature = self.property_embed(head_property_id_tensor_list[i])
            tail_property_feature = self.property_embed(tail_property_id_tensor_list[i])

            head_value_feature = head_property_value_embeds_matrix_tensor_list[i]
            tail_value_feature = tail_property_value_embeds_matrix_tensor_list[i]

            # here, it uses [0], to only use the first return value
            # (N, 1, D) * num_heads
            head_property_out = [g(head_value_feature, head_property_feature, rc_label_id)[0].unsqueeze(1) for g in
                                 self.head_att]
            # (N, 1, D) * num_heads
            tail_property_out = [g(tail_value_feature, tail_property_feature, rc_label_id)[0].unsqueeze(1) for g in
                                 self.tail_att]

            head_property_out = torch.cat(head_property_out, dim=1)  # (N, H, D)
            head_property_out = head_property_out.mean(dim=1)  # (768,)

            tail_property_out = torch.cat(tail_property_out, dim=1)
            tail_property_out = tail_property_out.mean(dim=1)  # (768,)

            # each is transformed by its own linear layer
            cat_out = torch.cat([head_property_out, tail_property_out], dim=0)

            cat_out = torch.unsqueeze(cat_out, dim=0)  # (1, 1536)

            score = self.lin1_list[rc_label_id](cat_out)  # (1, 1536) * (1536, 1) -> (1, 1)

            total_score.append(score)
        # endfor

        total_score = torch.cat(total_score, dim=1)
        total_score = torch.squeeze(total_score, dim=0)

        loss = None
        if nd_label_id_list is not None:
            # Note: get positive / negative logits here
            # positive
            pos_mask = nd_label_id_list.eq(1)
            pos_score = total_score[pos_mask]
            # negative
            neg_mask = nd_label_id_list.eq(0)
            neg_score = total_score[neg_mask]

            # Note: max margin loss
            max_margin_loss = nn.MarginRankingLoss(margin=1)
            target = torch.tensor([1] * int(batch_size / 2), dtype=torch.long).to(args.device)
            loss = max_margin_loss(pos_score, neg_score, target)
        # endif

        return total_score, loss


class RGAT_MaxMargin_Stack_With_Head_Tail_Attention_Info_Extraction(nn.Module):
    """
    relation-aware GAT
    
    extract attention formation

    The last linear transform layer is also stacked
    """

    def __init__(self, args, property_embed_tensor, num_of_rc_relation):
        """
        property_num: the total number of properties involved in one-hop of all our entities
        """
        super(RGAT_MaxMargin_Stack_With_Head_Tail_Attention_Info_Extraction, self).__init__()
        self.args = args

        args.embedding_dim = 768
        # num_embeddings, embed_dim = args.glove_embedding.shape
        # self.embed = nn.Embedding(num_embeddings, embed_dim)
        # self.embed.weight = nn.Parameter(
        #     args.glove_embedding, requires_grad=False)

        self.dropout = nn.Dropout(args.dropout)
        self.tanh = nn.Tanh()

        # if args.highway:
        #     self.highway_dep = Highway(args.num_layers, args.embedding_dim)
        #     self.highway = Highway(args.num_layers, args.embedding_dim)

        # self.bilstm = nn.LSTM(input_size=args.embedding_dim, hidden_size=args.hidden_size,
        #                       bidirectional=True, batch_first=True, num_layers=args.num_layers)
        # gcn_input_dim = args.hidden_size * 2

        # if args.gat:
        args.num_heads = 8
         # head Attention
        self.head_att = [
            RelationAttention(num_of_rc_relation=num_of_rc_relation, in_dim=args.embedding_dim).to(args.device) for i in
            range(args.num_heads)]
        
        # Tail Attention
        self.tail_att = [
            RelationAttention(num_of_rc_relation=num_of_rc_relation, in_dim=args.embedding_dim).to(args.device) for i in
            range(args.num_heads)]

        # if args.gat_attention_type == 'linear':
        #     self.gat = [LinearAttention(in_dim = gcn_input_dim, mem_dim = gcn_input_dim).to(args.device) for i in range(args.num_heads)] # we prefer to keep the dimension unchanged
        # elif args.gat_attention_type == 'dotprod':
        #     self.gat = [DotprodAttention().to(args.device) for i in range(args.num_heads)]
        # else:
        #     # reshaped gcn
        #     self.gat = nn.Linear(gcn_input_dim, gcn_input_dim)

        # property_embedding
        # self.property_embed = nn.Embedding(property_num, args.embedding_dim)
        # does not fine-tune during training, freeze=True

        self.property_embed = nn.Embedding.from_pretrained(property_embed_tensor, freeze=False)

        # tmp_embed = nn.Embedding(property_embed.size()[0], property_embed.size()[1])
        # nn.init.uniform_(tmp_embed.weight, -1.0, 1.0)
        # self.property_embed = tmp_embed

        # model.embedding.weight.grad
        # torch.manual_seed(3)
        # emb2 = nn.Embedding(5, 5)
        # nn.init.uniform_(emb2.weight, -1.0, 1.0)

        # last_hidden_size = args.hidden_size * 4
        #
        # layers = [
        #     nn.Linear(last_hidden_size, args.final_hidden_size), nn.ReLU()]
        # for _ in range(args.num_mlps-1):
        #     layers += [nn.Linear(args.final_hidden_size,
        #                          args.final_hidden_size), nn.ReLU()]
        # self.fcs = nn.Sequential(*layers)
        # self.fc_final = nn.Linear(args.final_hidden_size, args.num_classes)

        # self.lin1 = nn.Linear(768 * 2, 1)
        # glorot(self.lin1.weight)

        # last layer transform
        transform_linear_list = [nn.Linear(768 * 2, 1) for i in range(num_of_rc_relation)]

        # initialize
        for layer in transform_linear_list:
            glorot(layer.weight)
        # endfor

        self.lin1_list = torch.nn.ModuleList(transform_linear_list)

        pass

    def forward(self,
                args,
                head_property_id_tensor_list,
                tail_property_id_tensor_list,
                head_property_value_embeds_matrix_tensor_list,
                tail_property_value_embeds_matrix_tensor_list,
                nd_label_id_list,
                rc_label_id_list):
        '''
        Forward takes:
            sentence: sentence_id of size (batch_size, text_length)
            aspect: aspect_id of size (batch_size, aspect_length)
            pos_class: pos_tag_id of size (batch_size, text_length)
            dep_tags: dep_tag_id of size (batch_size, text_length)
            text_len: (batch_size,) length of each sentence
            aspect_len: (batch_size, ) aspect length of each sentence
            dep_rels: (batch_size, text_length) relation
            dep_heads: (batch_size, text_length) which node adjacent to that node
            aspect_position: (batch_size, text_length) mask, with the position of aspect as 1 and others as 0
            dep_dirs: (batch_size, text_length) the directions each node to the aspect
        '''
        # fmask = (torch.zeros_like(sentence) != sentence).float()  # (N，L)
        # dmask = (torch.zeros_like(dep_tags) != dep_tags).float()  # (N ,L)

        batch_size = len(head_property_id_tensor_list)

        total_score = []

        # ######## attention information #########
        head_attention_prob_mean_list = []
        tail_attention_prob_mean_list = []

        for i in range(len(rc_label_id_list)):
            
            # if cur_batch_data_guid_list[i] == "P161-4-test-s-1a8e70c1-6bed-421e-a696-adf8a6a7687e":
            #     print("hello")
            
            # relation_embed = self.property_embed(target_relation_id)
            rc_label_id = rc_label_id_list[i]

            head_property_feature = self.property_embed(head_property_id_tensor_list[i])
            tail_property_feature = self.property_embed(tail_property_id_tensor_list[i])

            head_value_feature = head_property_value_embeds_matrix_tensor_list[i]
            tail_value_feature = tail_property_value_embeds_matrix_tensor_list[i]

            # ######### (1) entity embed from RGAT #######
            # ######### (2) RGAT attention prob #######
            
            # # here, it uses [0], to only use the first return value
            # (N, 1, D) * num_heads
            # head_property_out = [g(head_value_feature, head_property_feature, rc_label_id)[0].unsqueeze(1) for g in
            #                      self.head_att]
            # # (N, 1, D) * num_heads
            # tail_property_out = [g(tail_value_feature, tail_property_feature, rc_label_id)[0].unsqueeze(1) for g in
            #                      self.tail_att]
            

            # #### head ####
            head_property_out_list = []
            head_att_prob_list = []
            for g in self.head_att:
                head_out, att_prob = g(head_value_feature, head_property_feature, rc_label_id)
                head_out = head_out.unsqueeze(1)
                head_property_out_list.append(head_out)
                head_att_prob_list.append(att_prob)
            # endfor

            # #### tail ####
            tail_property_out_list = []
            tail_att_prob_list = []
            for g in self.tail_att:
                tail_out, att_prob = g(tail_value_feature, tail_property_feature, rc_label_id)
                tail_out = tail_out.unsqueeze(1)
                tail_property_out_list.append(tail_out)
                tail_att_prob_list.append(att_prob)
            # endfor

            # # here, it uses [0], to only use the first return value
            # # (N, 1, D) * num_heads
            # head_property_out = [g(head_value_feature, head_property_feature, rc_label_id)[0].unsqueeze(1) for g in
            #                      self.gat_dep]
            # # (N, 1, D) * num_heads
            # tail_property_out = [g(tail_value_feature, tail_property_feature, rc_label_id)[0].unsqueeze(1) for g in
            #                      self.gat_dep]

            # #### take mean for entity embed #####
            head_property_out = torch.cat(head_property_out_list, dim=1)  # (N, H, D)
            head_property_out = head_property_out.mean(dim=1)  # (768,)

            tail_property_out = torch.cat(tail_property_out_list, dim=1)
            tail_property_out = tail_property_out.mean(dim=1)  # (768,)

            # ##### take mean for att prob ######
            head_att_prob = np.stack(head_att_prob_list)
            head_att_prob_mean = np.mean(head_att_prob, axis=0)
            head_attention_prob_mean_list.append(head_att_prob_mean)

            tail_att_prob = np.stack(tail_att_prob_list)
            tail_att_prob_mean = np.mean(tail_att_prob, axis=0)
            tail_attention_prob_mean_list.append(tail_att_prob_mean)

            # ######## calculate score ########
            # (1)
            cat_out = torch.cat([head_property_out, tail_property_out], dim=0)
            cat_out = torch.unsqueeze(cat_out, dim=0)  # (1, 1536)
            # (2) each is transformed by its own linear layer
            score = self.lin1_list[rc_label_id](cat_out)  # (1, 1536) * (1536, 1) -> (1, 1)
            total_score.append(score)
            # ###########

        # endfor

        total_score = torch.cat(total_score, dim=1)
        total_score = torch.squeeze(total_score, dim=0)

        loss = None
        if nd_label_id_list is not None:
            # Note: get positive / negative logits here
            # positive
            pos_mask = nd_label_id_list.eq(1)
            pos_score = total_score[pos_mask]
            # negative
            neg_mask = nd_label_id_list.eq(0)
            neg_score = total_score[neg_mask]

            # Note: max margin loss
            max_margin_loss = nn.MarginRankingLoss(margin=1)
            target = torch.tensor([1] * int(batch_size / 2), dtype=torch.long).to(args.device)
            loss = max_margin_loss(pos_score, neg_score, target)
        # endif

        return total_score, loss, head_attention_prob_mean_list, tail_attention_prob_mean_list



class UnsupervisedModel_Avg(torch.nn.Module):
    """
    1. Take the average of all words in each description
    2. Make a non-linear transformation of these two vector separately
      - non-linear transformation for subject
      - non-linear transformation for object
    3. loss function: make these difference between these two vectors small
    """

    def __init__(self, args):
        super(UnsupervisedModel_Avg, self).__init__()
        self.subj_linear = torch.nn.Linear(768, 768)
        self.obj_linear = torch.nn.Linear(768, 768)
        glorot(self.subj_linear.weight)
        glorot(self.obj_linear.weight)

    def forward(self, args, subject_embed_list, object_embed_list, label_id_list=None):
        loss = 0
        score_list = []
        for i in range(len(subject_embed_list)):
            subj_embed = subject_embed_list[i]
            obj_embed = object_embed_list[i]

            # took average of these two matrix
            subj_embed_avg = torch.mean(subj_embed, dim=0)
            obj_embed_avg = torch.mean(obj_embed, dim=0)

            subj_transform = torch.relu(self.subj_linear(subj_embed_avg))
            obj_transform = torch.relu(self.obj_linear(obj_embed_avg))

            cos = nn.CosineSimilarity(dim=0, eps=1e-6)
            tmp_loss = cos(subj_transform, obj_transform)

            score_list.append(tmp_loss.item())
            loss += tmp_loss
        # endfor

        return score_list, loss


class UnsupervisedModel_Interaction(torch.nn.Module):
    """
    Using unsupervised model, only train with the normal examples

    Interactive attention between subject and object
    The loss function is the minimize the distance between the two matrices
    """

    def __int__(self, args):
        super(UnsupervisedModel_Interaction, self).__int__()

    def forward(self, args, subject_embed_list, object_embed_list, label_id_list=None):
        score_list = []

        for i in range(len(subject_embed_list)):
            subj_embed = subject_embed_list[i]
            obj_embed = object_embed_list[i]

    pass


class SimpleInteractionModel(torch.nn.Module):
    """
    This is a supervised model, trained with contrastive examples

    subject_embed:  d * args.max_des
    object_embed:   d * args.max_des

    left_matrix x (subject_embed^T x object_embed) x right_matrix = \
                (1 * args.max_des) x (args.max_des * args.max_des) x (args.max * 1)
    """

    def __init__(self, args):
        super(SimpleInteractionModel, self).__init__()
        self.left_matrix = torch.nn.Linear(768, 1)
        self.right_matrix = torch.nn.Linear(768, 2)
        glorot(self.left_matrix.weight)
        glorot(self.right_matrix.weight)

    def forward(self, args, subject_embed_list, object_embed_list, label_id_list=None):

        score_list = []
        for i in range(len(subject_embed_list)):
            subj_embed = subject_embed_list[i]
            obj_embed = object_embed_list[i]

            interaction_matrix = torch.matmul(torch.transpose(subj_embed, 0, 1), obj_embed)
            left_result = self.left_matrix(interaction_matrix)
            score = self.right_matrix(torch.transpose(left_result, 0, 1))  # 1 x 2
            score_list.append(score)
        # endfor

        all_score = torch.cat(score_list)

        # loss
        if label_id_list is not None:
            loss_func = nn.CrossEntropyLoss()
            loss = loss_func(all_score, label_id_list)
            return all_score, loss
        else:
            return all_score, None


class Net(torch.nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()

        self.lin1 = torch.nn.Linear(args.input_size, args.hidden_size)
        self.convs = torch.nn.ModuleList()
        for i in range(args.stack_layer_num):
            self.convs.append(
                GATConv(args.hidden_size, args.hidden_size // args.heads, heads=args.heads, dropout=args.att_dropout))

        self.lin3 = torch.nn.Linear(args.hidden_size, 1)  # convert to one score
        # concatenate two object embeds together

        # self.rnn = torch.nn.LSTM(args.hidden_size, args.hidden_size, 1)
        glorot(self.lin3.weight)
        glorot(self.lin1.weight)

    # def forward(self, args, x, target_mask, edge_index, label_ids=None):
    def forward(self, args, word_embed_matrix, target_mask_list, graph_edge_list, label_id_list=None):

        # Note: calculate positive
        if args.embed_dropout > 0:
            word_embed_matrix = F.dropout(word_embed_matrix, p=args.embed_dropout, training=self.training)
        word_embed_matrix = self.lin1(word_embed_matrix)
        # output, (h, c) = self.rnn(torch.unsqueeze(x, 0))
        # x = torch.squeeze(h, 0)

        for i in range(args.stack_layer_num):
            word_embed_matrix = F.elu(self.convs[i](word_embed_matrix, graph_edge_list))
        # endfor

        # for i in range(args.stack_layer_num):
        #     output, (h, c) = self.rnn(torch.unsqueeze(F.elu(self.convs[i](x, edge_index)), 0), (h, c))
        #     x = torch.squeeze(h, 0)
        # # endfor

        batch_size = len(target_mask_list)

        target_embed_list = []
        for i in range(batch_size):
            t_mask_i = target_mask_list[i]
            target_embed = word_embed_matrix[t_mask_i]
            target_embed_list.append(target_embed)
        # endfor

        target_embed_list = torch.cat(target_embed_list, dim=0)

        logits = self.lin3(target_embed_list)
        total_score = torch.squeeze(logits, dim=1)
        # tanh = nn.Tanh()
        # logits = tanh(logits)  # add another

        loss = None
        if label_id_list is not None:
            # Note: get positive / negative logits here
            # positive
            pos_mask = label_id_list.eq(1)
            pos_score = total_score[pos_mask]
            # negative
            neg_mask = label_id_list.eq(0)
            neg_score = total_score[neg_mask]

            # Note: max margin loss
            max_margin_loss = nn.MarginRankingLoss(margin=1)
            target = torch.tensor([1] * int(batch_size / 2), dtype=torch.long).to(args.device)
            loss = max_margin_loss(pos_score, neg_score, target)
        # endif

        return total_score, loss
