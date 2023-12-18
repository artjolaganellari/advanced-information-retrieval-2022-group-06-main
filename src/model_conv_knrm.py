from typing import Dict, Iterator, List

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from allennlp.modules.text_field_embedders import TextFieldEmbedder

class Conv_KNRM(nn.Module):
    '''
    Paper: Convolutional Neural Networks for Soſt-Matching N-Grams in Ad-hoc Search, Dai et al. WSDM 18

    support taken from 
    https://github.com/sebastian-hofstaetter/matchmaker/blob/master/matchmaker/models/conv_knrm.py
    '''

    def __init__(self,
                 word_embeddings: TextFieldEmbedder,
                 n_grams:int,
                 n_kernels: int,
                 conv_out_dim:int):

        super(Conv_KNRM, self).__init__()

        self.word_embeddings = word_embeddings

        # static - kernel size & magnitude variables
        mu = torch.FloatTensor(self.kernel_mus(n_kernels)).view(1, 1, 1, n_kernels)
        sigma = torch.FloatTensor(self.kernel_sigmas(n_kernels)).view(1, 1, 1, n_kernels)

        self.register_buffer('mu', mu) #can be accessed via self.mu
        self.register_buffer('sigma', sigma) #can be accessed via self.sigma

        # todo
        # conv-knrm considers convolutions of kernel size 1 to n_grams (i.e. considers single words to n_grams)        
        # we first create this list of of 1-D convolutions. The kernel of each convolution corresponds to the
        # number of N-grams that we want to consider in the convolution. N goes from 1 (i.e. singel words) up to n_grams        
        self.convolutions = []
        for i in range(1, n_grams + 1):
            self.convolutions.append(
                nn.Sequential(
                    nn.ConstantPad1d((0,i - 1), 0), #only pad at the end of the sequence, do not use padding at the beginning
                    nn.Conv1d(kernel_size=i, in_channels=word_embeddings.get_output_dim(), out_channels=conv_out_dim), #kernel_size=number of grams to consider, in_channels is given by the TextFieldEmbedder
                    nn.ReLU()) #only consider positive activations
            )
        #we register all convolutions in an nn.ModuleList
        self.convolutions = nn.ModuleList(self.convolutions) 

        # create the final linear layer for scoring that takes Kernel-Matrix as input and outputs a 1-D score
        # for each kernel and for each N-gram pair of document and query, we get an input for the linear layer.
        # hence, the input dimension therefore amounts to n_kernels x n_grams x n_grams
        # (in the lecture notes, the bias term is grayed out. We believe that it doesn't harm if we include it)
        # (different decision than in https://github.com/sebastian-hofstaetter/matchmaker/blob/master/matchmaker/models/knrm.py)
        self.dense = nn.Linear(n_kernels * n_grams * n_grams, 1) 

        # we initialise the linear layer with small weights (according to https://github.com/sebastian-hofstaetter/matchmaker/blob/master/matchmaker/models/conv_knrm.py)
        torch.nn.init.uniform_(self.dense.weight, -0.014, 0.014)  # inits taken from matchzoo

    def forward(self, query: Dict[str, torch.Tensor], document: Dict[str, torch.Tensor]) -> torch.Tensor:

        #
        # prepare embedding tensors
        # -------------------------------------------------------

        # we assume 0 is padding - both need to be removed
        # shape: (batch, query_max)
        query_pad_mask = (query["tokens"]["tokens"] > 0).float() # > 1 to also mask oov terms
        # shape: (batch, doc_max)
        document_pad_mask = (document["tokens"]["tokens"] > 0).float()

        # shape: (batch, query_max,emb_dim)
        query_embeddings = self.word_embeddings(query)
        # shape: (batch, document_max,emb_dim)
        document_embeddings = self.word_embeddings(document)

        #todo
        #we have to ignore all query and document entries that are "artificially" made by paddings. Therefore we matrix-multiply
        #query_pad_mask (only True or 1 for no-padding entries) with the transposed matrix of document_pad_mask
        query_by_doc_mask = torch.bmm(query_pad_mask.unsqueeze(-1), document_pad_mask.unsqueeze(-1).transpose(-1, -2))

        # !! conv1d requires tensor in shape: [batch, emb_dim, sequence_length ]
        # so we transpose embedding tensors from : [batch, sequence_length,emb_dim] to [batch, emb_dim, sequence_length ]
        # feed that into the conv1d and reshape output from [batch, conv1d_out_channels, sequence_length ] to [batch, sequence_length, conv1d_out_channels]

        # re-shape to [batch, emb_dim, sequence_length ]
        query_embeddings_t = query_embeddings.transpose(1, 2)
        document_embeddings_t = document_embeddings.transpose(1, 2)

        query_results = []
        document_results = []

        #apply convolution of nn.ModuleList (each convolution corresponds to N-gram applied to query and document that we analyse)
        for i,conv in enumerate(self.convolutions):
            query_conv = conv(query_embeddings_t).transpose(1, 2) #reshape results back to [batch, sequence_length, conv1d_out_channels]
            document_conv = conv(document_embeddings_t).transpose(1, 2) #reshape results back to [batch, sequence_length, conv1d_out_channels]

            query_results.append(query_conv)
            document_results.append(document_conv)


        #we now match the convoluted query tokens with the convoluted document tokens according to all possible combinations
        matched_results = []
        for i in range(len(query_results)): #for all CNN-N-gram representations of the query
            for t in range(len(document_results)): #for all CNN-N-gram representations of the document

                ###########################################################
                ### Consine similarity between query and document embedding
                ###########################################################                
                query_norm = query_results[i] / (query_results[i].norm(p=2, dim=-1, keepdim=True) + 1e-13)
                document_norm = document_results[t] / (document_results[t].norm(p=2, dim=-1, keepdim=True) + 1e-13)

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
                kernel_results_masked = raw_kernel_results * query_by_doc_mask.unsqueeze(-1)

                #for each kernel and each query token, sum the kernel values at all document tokens
                per_kernel_query = torch.sum(kernel_results_masked, 2)
                #for each kernel and query token, take the natural logarithm of per_kernel_query. As very small (e.g. previously masked) per_kernel_query entries would give
                # a logarithm of -np.Inf, clamp per_kernel_query entries at 10**(-10). 
                log_per_kernel_query = torch.log(torch.clamp(per_kernel_query, min=1e-10)) * 0.01                
                #Furthermore, set the result for previously masked per_kernel_query terms to 0,
                # as we do not want to let padded terms influence the result
                log_per_kernel_query_masked = log_per_kernel_query * query_pad_mask.unsqueeze(-1) # make sure we mask out padding values

                #for each kernel, sum over masked kernel values at all query tokens
                per_kernel = torch.sum(log_per_kernel_query_masked, 1) 
                
                #append the matched results
                matched_results.append(per_kernel)

        
        # After determining the kernel values of the matching matrix (i.e. summing how good/ bad document and query-CNN-N-grams match)
        # we feed everything into the final linear layer to predict the score

        #concatenate all matched_results into one tensor all_grams
        all_grams = torch.cat(matched_results,1)

        #feed all_grams into the final linear layer to predict the score
        dense_out = self.dense(all_grams)        

        output = torch.squeeze(dense_out, 1)        

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
        l_sigma = [0.001]  # for exact match. small variance -> exact match
        if n_kernels == 1:
            return l_sigma

        l_sigma += [0.5 * bin_size] * (n_kernels - 1)
        return l_sigma