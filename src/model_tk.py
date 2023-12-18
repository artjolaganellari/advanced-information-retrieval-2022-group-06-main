from typing import Dict, Iterator, List

import torch
import torch.nn as nn
from torch.autograd import Variable

from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.matrix_attention.cosine_matrix_attention import CosineMatrixAttention 

class TK(nn.Module):
    '''
    Paper: S. Hofstätter, M. Zlabinger, and A. Hanbury 2020. Interpretable & Time-Budget-Constrained Contextualization for Re-Ranking. In Proc. of ECAI 
    '''


    def __init__(self,
                 word_embeddings: TextFieldEmbedder,
                 n_kernels: int):

        super(TK, self).__init__()

        self.word_embeddings = word_embeddings

        # static - kernel size & magnitude variables
        if torch.cuda.is_available():
            self.mu = Variable(torch.cuda.FloatTensor(self.kernel_mus(n_kernels)), requires_grad=False) \
                .view(1, 1, 1, n_kernels)
            self.sigma = Variable(torch.cuda.FloatTensor(self.kernel_sigmas(n_kernels)), requires_grad=False) \
                .view(1, 1, 1, n_kernels)
        else:
            self.mu = Variable(torch.FloatTensor(self.kernel_mus(n_kernels)), requires_grad=False) \
                .view(1, 1, 1, n_kernels)
            self.sigma = Variable(torch.FloatTensor(self.kernel_sigmas(n_kernels)), requires_grad=False) \
                .view(1, 1, 1, n_kernels)

        #todo
        self.cosine_module = CosineMatrixAttention()
        self.nn_scaler = nn.Parameter(torch.full([1], 0.01, dtype=torch.float32, requires_grad=True))

        self.dense = nn.Linear(n_kernels, 1, bias=False)
        self.dense_mean = nn.Linear(n_kernels, 1, bias=False)
        self.dense_comb = nn.Linear(2, 1, bias=False)

        # init with small weights, otherwise the dense output is way to high for the tanh -> resulting in loss == 1 all the time
        torch.nn.init.uniform_(self.dense.weight, -0.014, 0.014)  # inits taken from matchzoo
        torch.nn.init.uniform_(self.dense_mean.weight, -0.014, 0.014)  # inits taken from matchzoo

        # init with small weights, otherwise the dense output is way to high for the tanh -> resulting in loss == 1 all the time
        torch.nn.init.uniform_(self.dense.weight, -0.014, 0.014)  # inits taken from matchzoo

    def forward(self, query: Dict[str, torch.Tensor], document: Dict[str, torch.Tensor]) -> torch.Tensor:

        # shape: (batch, query_max)
        query_pad_oov_mask = (query["tokens"]["tokens"] > 0).float() # > 1 to also mask oov terms
        # shape: (batch, doc_max)
        document_pad_oov_mask = (document["tokens"]["tokens"] > 0).float()

        # shape: (batch, query_max,emb_dim)
        query_embeddings = self.word_embeddings(query)
        # shape: (batch, document_max,emb_dim)
        document_embeddings = self.word_embeddings(document)

        #todo
        # prepare embedding tensors & paddings masks
        # -------------------------------------------------------

        query_by_doc_mask = torch.bmm(query_pad_oov_mask.unsqueeze(-1), document_pad_oov_mask.unsqueeze(-1).transpose(-1, -2))
        query_by_doc_mask_view = query_by_doc_mask.unsqueeze(-1)

        # shape: (batch, query_max, doc_max)
        cosine_matrix = self.cosine_module.forward(query_embeddings, document_embeddings)
        cosine_matrix_masked = cosine_matrix * query_by_doc_mask
        cosine_matrix_extradim = cosine_matrix_masked.unsqueeze(-1)

        #
        # gaussian kernels & soft-TF
        #
        # first run through kernel, then sum on doc dim then sum on query dim
        # -------------------------------------------------------
        
        raw_kernel_results = torch.exp(- torch.pow(cosine_matrix_extradim - self.mu, 2) / (2 * torch.pow(self.sigma, 2)))
        kernel_results_masked = raw_kernel_results * query_by_doc_mask_view

        #
        # mean kernels
        #
        #kernel_results_masked2 = kernel_results_masked.clone()

        doc_lengths = torch.sum(document_pad_oov_mask, 1)

        #kernel_results_masked2_mean = kernel_results_masked / doc_lengths.unsqueeze(-1)

        per_kernel_query = torch.sum(kernel_results_masked, 2)
        log_per_kernel_query = torch.log2(torch.clamp(per_kernel_query, min=1e-10)) * self.nn_scaler
        log_per_kernel_query_masked = log_per_kernel_query * query_pad_oov_mask.unsqueeze(-1) # make sure we mask out padding values
        per_kernel = torch.sum(log_per_kernel_query_masked, 1) 

        per_kernel_query_mean = per_kernel_query / (doc_lengths.view(-1,1,1) + 1) # well, that +1 needs an explanation, sometimes training data is just broken ... (and nans all the things!)

        log_per_kernel_query_mean = per_kernel_query_mean * self.nn_scaler
        log_per_kernel_query_masked_mean = log_per_kernel_query_mean * query_pad_oov_mask.unsqueeze(-1) # make sure we mask out padding values
        per_kernel_mean = torch.sum(log_per_kernel_query_masked_mean, 1) 

        ##
        ## "Learning to rank" layer - connects kernels with learned weights
        ## -------------------------------------------------------

        dense_out = self.dense(per_kernel)
        dense_mean_out = self.dense_mean(per_kernel_mean)
        dense_comb_out = self.dense_comb(torch.cat([dense_out,dense_mean_out],dim=1))
        output = torch.squeeze(dense_comb_out,1) #torch.tanh(dense_out), 1)

        return output

    def kernel_mus(self, n_kernels: int):
        """
        get the mu for each guassian kernel. Mu is the middle of each bin
        :param n_kernels: number of kernels (including exact match). first one is exact match
        :return: l_mu, a list of mu.
        """
        l_mu = [1.0]
        if n_kernels == 1:
            return l_mu

        bin_size = 2.0 / (n_kernels - 1)  # score range from [-1, 1]
        l_mu.append(1 - bin_size / 2)  # mu: middle of the bin
        for i in range(1, n_kernels - 1):
            l_mu.append(l_mu[i] - bin_size)
        return l_mu

    def kernel_sigmas(self, n_kernels: int):
        """
        get sigmas for each guassian kernel.
        :param n_kernels: number of kernels (including exactmath.)
        :param lamb:
        :param use_exact:
        :return: l_sigma, a list of simga
        """
        bin_size = 2.0 / (n_kernels - 1)
        l_sigma = [0.0001]  # for exact match. small variance -> exact match
        if n_kernels == 1:
            return l_sigma

        l_sigma += [0.5 * bin_size] * (n_kernels - 1)
        return l_sigma
