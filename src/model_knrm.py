from typing import Dict, Iterator, List

import torch
import torch.nn as nn
from torch.autograd import Variable

from allennlp.modules.text_field_embedders import TextFieldEmbedder


class KNRM(nn.Module):
    '''
    Paper: End-to-End Neural Ad-hoc Ranking with Kernel Pooling, Xiong et al., SIGIR'17

    support taken from 
    https://github.com/sebastian-hofstaetter/matchmaker/blob/master/matchmaker/models/knrm.py
    '''

    def __init__(self,
                 word_embeddings: TextFieldEmbedder,
                 n_kernels: int):

        super(KNRM, self).__init__()

        self.word_embeddings = word_embeddings

        # static - kernel size & magnitude variables
        mu = torch.FloatTensor(self.kernel_mus(n_kernels)).view(1, 1, 1, n_kernels)
        sigma = torch.FloatTensor(self.kernel_sigmas(n_kernels)).view(1, 1, 1, n_kernels)

        #use mu and sigmas as constants (without gradient)
        self.register_buffer('mu', mu)  #can be accessed via self.mu
        self.register_buffer('sigma', sigma) #can be accessed via self.sigma

        #todo
        #we include a fully connected linear layer with bias to calculat the score based on the kernel-values
        # (in the lecture notes, the bias term is grayed out. We believe that it doesn't harm if we include it)
        # (different decision than in https://github.com/sebastian-hofstaetter/matchmaker/blob/master/matchmaker/models/knrm.py)
        self.linear = nn.Linear(n_kernels, 1)

    def forward(self, query: Dict[str, torch.Tensor], document: Dict[str, torch.Tensor]) -> torch.Tensor:
        # pylint: disable=arguments-differ

        #
        # prepare embedding tensors & paddings masks
        # -------------------------------------------------------

        # shape: (batch, query_max)
        query_pad_oov_mask = (query["tokens"]["tokens"] > 0).float() # > 1 to also mask oov terms
        # shape: (batch, doc_max)
        document_pad_oov_mask = (document["tokens"]["tokens"] > 0).float()

        # shape: (batch, query_max,emb_dim)
        query_embeddings = self.word_embeddings(query)
        # shape: (batch, document_max,emb_dim)
        document_embeddings = self.word_embeddings(document)

        #todo

        #we have to ignore all query and document entries that are "artificially" made by paddings. Therefore we matrix-multiply
        #query_pad_oov_mask (only True or 1 for no-padding entries) with the transposed matrix of document_pad_oov_mask
        query_by_doc_mask = torch.bmm(query_pad_oov_mask.unsqueeze(-1), document_pad_oov_mask.unsqueeze(-1).transpose(-1, -2))
        query_by_doc_mask_view = query_by_doc_mask.unsqueeze(-1)

        ###########################################################
        ### Consine similarity between query and document embedding
        ###########################################################

        query_norm = query_embeddings / (query_embeddings.norm(p=2, dim=-1, keepdim=True) + 1e-13)
        document_norm = document_embeddings / (document_embeddings.norm(p=2, dim=-1, keepdim=True) + 1e-13)

        #calculate cosine similarity as matrix-product of query_norm and (transposed) document_norm
        cosine_matrix = torch.bmm(query_norm, document_norm.transpose(-1, -2)) 
        #ignore all consine similarity matrix entries that are due to padding entries of query or document tokens
        cosine_matrix_masked = cosine_matrix * query_by_doc_mask
        #include an extra dimension at the end of the tensor
        cosine_matrix_extradim = cosine_matrix_masked.unsqueeze(-1)

        ###########################################################
        #### Kernel values based on cosine_similarity matrix
        ###########################################################

        #for each kernel, calculate the kernel values w.r.t. mus and sigmas at the cosine similarity entries
        raw_kernel_results = torch.exp(- torch.pow(cosine_matrix_extradim - self.mu, 2) / (2 * torch.pow(self.sigma, 2)))

        #ignore kernel values of cosine_matrix-entries that are due to padding entries of query or document tokens.
        kernel_results_masked = raw_kernel_results * query_by_doc_mask_view

        #for each kernel and each query token, sum the kernel values at all document tokens
        per_kernel_query = torch.sum(kernel_results_masked, 2)

        #for each kernel and query token, take the natural logarithm of per_kernel_query. As very small (e.g. previously masked) per_kernel_query entries would give
        # a logarithm of -np.Inf, clamp per_kernel_query entries at 10**(-10). Furthermore, set the result for previously masked per_kernel_query terms to 0,
        # as we do not want to let padded terms influence the result
        log_per_kernel_query_masked = torch.log(torch.clamp(per_kernel_query, min=10**(-10))) * query_pad_oov_mask.unsqueeze(-1)
        
        #for each kernel, sum over masked kernel values at all query tokens
        per_kernel = torch.sum(log_per_kernel_query_masked, 1) 

        ###########################################################
        #calculate score based on per_kernel values 
        #using the linear connected layer self.linear 
        ###########################################################

        linear_out = self.linear(per_kernel)
        output = torch.squeeze(linear_out,1) #remove dimensions of size 1
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
