"""
In this file, you should implement the forward calculation of the conventional RNN, GRU, and Multi-Head Attention (MHA). 
Please use the provided interface. The arguments are explained in the documentation of the three functions.

You also need to implement two functions that configurate GRUs in special ways.
"""
import torch
import numpy as np
from scipy.special import expit as sigmoid

def rnn(wt_h, wt_x, bias, init_state, input_data):
    """
    RNN forward calculation.

    args:
        wt_h: shape [hidden_size, hidden_size], weight matrix for hidden state transformation. Rows corresponds 
              to dimensions of previous hidden states
        wt_x: shape [input_size, hidden_size], weight matrix for input transformation
        bias: shape [hidden_size], bias term
        init_state: shape [hidden_size], the initial state of the RNN
        input_data: shape [batch_size, time_steps, input_size], input data of `batch_size` sequences, each of
                    which has length `time_steps` and `input_size` features at each time step. 
    returns:
        outputs: shape [batch_size, time_steps, hidden_size], outputs along the sequence. The output at each 
                 time step is exactly the hidden state
        final_state: the final hidden state
    """

    # TODO: loop over time steps

    # TODO: implement the RNN calculation at each step
    
    outputs = list()
    state = init_state[0]
    for i in range(input_data.shape[1]):
        state = np.tanh(input_data[:,i,:]@wt_x+state@wt_h+bias)
        outputs.append(state)
    return np.swapaxes(outputs,0,1), np.array([state])


def gru(linear_trans_r, linear_trans_z, linear_trans_n, init_state, input_data):
    """
    GRU forward calculation. NOTE: please use the calculation in the documentation of `torch.nn.GRU`, not the 
    formulation from the d2l book. 

    args:
        linear_trans_r: linear transformation weights and biases for the R gate
        linear_trans_z: linear transformation weights and biases for the Z gate
        linear_trans_n: linear transformation weights and biases for the candidate hidden state
        init_state: shape [hidden_size], the initial state of the RNN
        input_data: shape [batch_size, time_steps, input_size], input data of `batch_size` sequences, each of
                    which has length `time_steps` and `input_size` features at each time step. 
    returns:
        outputs: shape [batch_size, time_steps, hidden_size], outputs along the sequence. The output at each 
                 time step is exactly the hidden state
        final_state: the final hidden state
    """

    # unpack weights/biases from the three arguments
    (wt_ir, biasir, wt_hr, biashr) = linear_trans_r
    (wt_iz, biasiz, wt_hz, biashz) = linear_trans_z 
    (wt_in, biasin, wt_hn, biashn) = linear_trans_n
 
    # TODO: loop over time steps

    # TODO: compute Z gate

    # TODO: compute R gate

    # TODO: compute candiate hidden state using the R gate
   
    # TODO: compute the final output
        
    outputs = list()
    state = init_state[0]
    for i in range(input_data.shape[1]):
        Z = sigmoid(input_data[:,i,:]@wt_iz+biasiz+state@wt_hz+biashz)
        R = sigmoid(input_data[:,i,:]@wt_ir+biasir+state@wt_hr+biashr)
        state_tilde = np.tanh(input_data[:,i,:]@wt_in+biasin+R*(state@wt_hn+biashn))
        state = Z * state + (1 - Z) * state_tilde
        outputs.append(state)
    return np.swapaxes(outputs,0,1), np.array([state])



def init_gru_with_rnn(wt_h, wt_x, bias):
    """
    This function compute parameters of a GRU such that it performs like a conventional RNN. The input are parameters 
    of an RNN, and the parameters of the GRU are returned. Please use the same format as `rnn_param_helper.get_gru_params`
    when returning weights for the GRU.  

    args:
        wt_h: shape [hidden_size, hidden_size], weight matrix for hidden state transformation. Rows corresponds 
              to dimensions of previous hidden states
        wt_x: shape [input_size, hidden_size], weight matrix for input transformation
        bias: shape [hidden_size], bias term

    returns:
        linear_trans_r: linear transformation weights and biases for the R gate
        linear_trans_z: linear transformation weights and biases for the Z gate
        linear_trans_n: linear transformation weights and biases for the candidate hidden state
    """

    # TODO: Set the linear transformation for the R gate

    wt_ir  = np.zeros(wt_x.shape)
    wt_hr  = np.zeros(wt_h.shape)
    biasir = np.ones(bias.shape) * 1e10
    biashr = np.ones(bias.shape) * 1e10


    # TODO: Set the linear transformation for the Z gate

    wt_iz  = np.zeros(wt_x.shape)
    wt_hz  = np.zeros(wt_h.shape)
    biasiz = np.ones(bias.shape) * -1e10
    biashz = np.ones(bias.shape) * -1e10


    # TODO:  Set the linear transformation for the candidate hidden state 

    wt_in  = wt_x 
    wt_hn  = wt_h 
    biasin = bias/2
    biashn = bias/2
     
    linear_trans_r = (wt_ir, biasir, wt_hr, biashr)
    linear_trans_z = (wt_iz, biasiz, wt_hz, biashz)
    linear_trans_n = (wt_in, biasin, wt_hn, biashn)

    return linear_trans_r, linear_trans_z, linear_trans_n


def init_gru_with_long_term_memory(input_size, hidden_size):
    """
    This function compute parameters of a GRU such that it maintains the initial state in the memory. 
    Please use the same format as `rnn_param_helper.get_gru_params` when returning weights for the GRU.  

    args:
        input_size: int, the input dimension 
        hidden_size: int, the hidden dimension

    returns:
        linear_trans_r: linear transformation weights and biases for the R gate
        linear_trans_z: linear transformation weights and biases for the Z gate
        linear_trans_n: linear transformation weights and biases for the candidate hidden state
    """

    # TODO: Set the linear transformation for the R gate

    # should not matter what R is
    wt_ir  = np.zeros((input_size,hidden_size))
    wt_hr  = np.zeros((hidden_size,hidden_size))
    biasir = np.zeros((hidden_size,))
    biashr = np.zeros((hidden_size,))

    # TODO: Set the linear transformation for the Z gate

    wt_iz  = np.zeros((input_size,hidden_size))
    wt_hz  = np.zeros((hidden_size,hidden_size))
    biasiz = np.ones((hidden_size,)) * 1e10
    biashz = np.ones((hidden_size,)) * 1e10


    # TODO:  Set the linear transformation for the candidate hidden state 

    # should not matter what candidate hidden state is
    wt_in  = np.zeros((input_size,hidden_size)) 
    wt_hn  = np.zeros((hidden_size,hidden_size))
    biasin = np.zeros((hidden_size,))
    biashn = np.zeros((hidden_size,))
     
    linear_trans_r = (wt_ir, biasir, wt_hr, biashr)
    linear_trans_z = (wt_iz, biasiz, wt_hz, biashz)
    linear_trans_n = (wt_in, biasin, wt_hn, biashn)

    return linear_trans_r, linear_trans_z, linear_trans_n


def mha(Wq, Wk, Wv, Wo, input_data):
    """
    This function implements torch MHA layer with the following configuration.  

    `nn.MultiheadAttention(embed_dim=input_size, num_heads=num_heads, dropout=0.0, bias=False, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None, batch_first=True)`

    The calculation is defined by the two equations in the documentation of `nn.MultiheadAttention`. 

    args:
        Wq: a list of matrices, each of which has shape [embed_dim, embed_dim // num_head]
        Wk: a list of matrices, each of which has shape [embed_dim, embed_dim // num_head]
        Wv: a list of matrices, each of which has shape [embed_dim, embed_dim // num_head]
        Wo: a numpy tensor with shape [embed_dim, embed_dim]
        input_data: a tensor with shape [batch_size, sequence_length, embed_dim]. Note that we have the `batch_first` flag on, so the first dimension corresponding to 
                    the batch dimension

    returns:

        output: a tensor with shape [batch_size, sequence_length, embed_dim]

    """

    # TODO: using the three matrices to transform X into Q, K V               

    # TODO: compute the attention matrix from Q, K
                
    # TODO: compute the weighted sum of rows in V

    # TODO: concatenate and do the output transformation 

    d, n = input_data.shape[-1], len(Wq)
    output = list()
    for k in range(n):
        Q = input_data@Wq[k]
        K = input_data@Wk[k]
        V = input_data@Wv[k]
        QKT = softmax(torch.bmm(Q, torch.transpose(K, 1, 2))/np.sqrt(d/n))
        output.append(QKT@V)
    output = np.concatenate(output, axis=2)
    return output@Wo

def softmax(x, dim=-1):
    return torch.nn.functional.softmax(x, dim=dim)

