
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from param import args
import numpy as np

def check(ar):
    ar = ar.cpu().detach().numpy()
    return np.any(np.isnan(ar))

def check2(ar):
    # ar = ar.cpu().numpy()
    return np.any(np.isnan(ar))

import utils
TRAIN_VOCAB = '../tasks/R2R/data/train_vocab.txt'
vocab = utils.read_vocab(TRAIN_VOCAB)
tok = utils.Tokenizer(vocab=vocab, encoding_length=args.maxInput)

#
class EncoderLSTM(nn.Module):
    ''' Encodes navigation instructions, returning hidden state context (for
        attention methods) and a decoder initial state. '''

    def __init__(self, vocab_size, embedding_size, hidden_size, padding_idx, 
                            dropout_ratio, bidirectional=False, num_layers=1):
        super(EncoderLSTM, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.drop = nn.Dropout(p=dropout_ratio)
        if bidirectional:
            print("Using Bidir in EncoderLSTM")
        self.num_directions = 2 if bidirectional else 1
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx)
        input_size = embedding_size
        self.lstm = nn.LSTM(input_size, hidden_size, self.num_layers,
                            batch_first=True, dropout=dropout_ratio, 
                            bidirectional=bidirectional)
        self.encoder2decoder = nn.Linear(hidden_size * self.num_directions,
            hidden_size * self.num_directions
        )

    def init_state(self, inputs):
        ''' Initialize to zero cell states and hidden states.'''
        batch_size = inputs.size(0)
        h0 = Variable(torch.zeros(
            self.num_layers * self.num_directions,
            batch_size,
            self.hidden_size
        ), requires_grad=False)
        c0 = Variable(torch.zeros(
            self.num_layers * self.num_directions,
            batch_size,
            self.hidden_size
        ), requires_grad=False)

        return h0.cuda(), c0.cuda()

    def forward(self, inputs, lengths):
        ''' Expects input vocab indices as (batch, seq_len). Also requires a 
            list of lengths for dynamic batching. '''
        embeds = self.embedding(inputs)  # (batch, seq_len, embedding_size)
        if check(embeds):
            print('embeds before drop')
            print(embeds)
        embeds = self.drop(embeds)
        h0, c0 = self.init_state(inputs)
        packed_embeds = pack_padded_sequence(embeds, lengths, batch_first=True)
        enc_h, (enc_h_t, enc_c_t) = self.lstm(packed_embeds, (h0, c0))
        
        if check(embeds):
            print('embeds')
            print(embeds)
            embeds_np = embeds.cpu().detach().numpy()
            inputs_np = inputs.cpu().detach().numpy()
            for i,sen in enumerate(embeds_np):
                sen_o = inputs_np[i]
                for j,v in enumerate(sen):
                    v_o = sen_o[j]
                    if check2(v):
                        print('sentence',i,'vab',j,'id',v_o,tok.index_to_word[v_o])
            
        
        # if check2(enc_h):
        #     print('enc_h')
        #     print(enc_h)

        # if check2(enc_h_t):
        #     print('enc_h_t')
        #     print(enc_h_t)
        
        # if check2(enc_c_t):
        #     print('enc_c_t')
        #     print(enc_c_t)

        if self.num_directions == 2:    # The size of enc_h_t is (num_layers * num_directions, batch, hidden_size)
            h_t = torch.cat((enc_h_t[-1], enc_h_t[-2]), 1)
            c_t = torch.cat((enc_c_t[-1], enc_c_t[-2]), 1)
        else:
            h_t = enc_h_t[-1]
            c_t = enc_c_t[-1] # (batch, hidden_size)

        ctx, _ = pad_packed_sequence(enc_h, batch_first=True)

        if args.sub_out == "max":
            ctx_max, _ = ctx.max(1)
            decoder_init = nn.Tanh()(self.encoder2decoder(ctx_max))
        elif args.sub_out == "tanh":
            decoder_init = nn.Tanh()(self.encoder2decoder(h_t))
        else:
            assert False

        ctx = self.drop(ctx)
        if args.zero_init:
            return ctx, torch.zeros_like(decoder_init), torch.zeros_like(c_t)
        else:
            return ctx, decoder_init, c_t  # (batch, seq_len, hidden_size*num_directions)
                                 # (batch, hidden_size)

#
class SoftDotAttention(nn.Module):
    '''Soft Dot Attention. 

    Ref: http://www.aclweb.org/anthology/D15-1166
    Adapted from PyTorch OPEN NMT.
    '''

    def __init__(self, query_dim, ctx_dim):
        '''Initialize layer.'''
        super(SoftDotAttention, self).__init__()
        self.linear_in = nn.Linear(query_dim, ctx_dim, bias=False)
        self.sm = nn.Softmax()
        self.linear_out = nn.Linear(query_dim + ctx_dim, query_dim, bias=False)
        self.tanh = nn.Tanh()

    def forward(self, h, context, mask=None,
                output_tilde=True, output_prob=True):
        '''Propagate h through the network.

        h: batch x dim
        context: batch x seq_len x dim
        mask: batch x seq_len indices to be masked
        '''
        target = self.linear_in(h).unsqueeze(2)  # batch x dim x 1
        # print(context.shape, target.shape)
        # Get attention
        attn = torch.bmm(context, target).squeeze(2)  # batch x seq_len
        logit = attn

        if mask is not None:
            # -Inf masking prior to the softmax
            attn.masked_fill_(mask, -float('inf'))
        attn = self.sm(attn)    # There will be a bug here, but it's actually a problem in torch source code.
        attn3 = attn.view(attn.size(0), 1, attn.size(1))  # batch x 1 x seq_len

        weighted_context = torch.bmm(attn3, context).squeeze(1)  # batch x dim
        if not output_prob:
            attn = logit
        if output_tilde:
            h_tilde = torch.cat((weighted_context, h), 1)
            h_tilde = self.tanh(self.linear_out(h_tilde))
            return h_tilde, weighted_context, attn
        else:
            return weighted_context, attn


#
class AttnDecoderLSTM(nn.Module):
    ''' An unrolled LSTM with attention over instructions for decoding navigation actions. '''

    def __init__(self, embedding_size, hidden_size,
                       dropout_ratio, feature_size=2048+4):
        super(AttnDecoderLSTM, self).__init__()
        self.embedding_size = embedding_size
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.embedding = nn.Sequential(
            nn.Linear(args.angle_feat_size, self.embedding_size),
            nn.Tanh()
        )
        self.drop = nn.Dropout(p=dropout_ratio)
        self.drop_env = nn.Dropout(p=args.featdropout)
        self.lstm = nn.LSTMCell(embedding_size+feature_size, hidden_size)
        self.feat_att_layer = SoftDotAttention(hidden_size, feature_size)
        self.attention_layer = SoftDotAttention(hidden_size, hidden_size)
        self.candidate_att_layer = SoftDotAttention(hidden_size, feature_size)

    def forward(self, action, feature, cand_feat,
                h_0, prev_h1, c_0,
                ctx, ctx_mask=None,
                already_dropfeat=False,
                h_1_res=None,
                cand_flag=False
                ):
        '''
        Takes a single step in the decoder LSTM (allowing sampling).
        action: batch x angle_feat_size
        feature: batch x 36 x (feature_size + angle_feat_size)
        cand_feat: batch x cand x (feature_size + angle_feat_size)
        h_0: batch x hidden_size
        prev_h1: batch x hidden_size
        c_0: batch x hidden_size
        ctx: batch x seq_len x dim
        ctx_mask: batch x seq_len - indices to be masked
        already_dropfeat: used in EnvDrop
        h_1_res: batch x hidden_size
        '''
        action_embeds = self.embedding(action)

        # Adding Dropout
        action_embeds = self.drop(action_embeds)

        # if not already_dropfeat:
        #     # Dropout the raw feature as a common regularization
        #     feature[..., :-args.angle_feat_size] = self.drop_env(feature[..., :-args.angle_feat_size])   # Do not drop the last args.angle_feat_size (position feat)
        if not already_dropfeat:
            # Dropout the raw feature as a common regularization
            f = self.drop_env(feature[..., :-args.angle_feat_size])
            a = feature[..., -args.angle_feat_size:]
            feature = torch.cat([f, a],-1)
            # feature[..., :-args.angle_feat_size] = self.drop_env(feature[..., :-args.angle_feat_size])   # Do not drop the last args.angle_feat_size (position feat)



        prev_h1_drop = self.drop(prev_h1)
        attn_feat, _ = self.feat_att_layer(prev_h1_drop, feature, output_tilde=False)

        concat_input = torch.cat((action_embeds, attn_feat), 1) # (batch, embedding_size+feature_size)
        h_1, c_1 = self.lstm(concat_input, (prev_h1, c_0))

        h_1_drop = self.drop(h_1)
        h_tilde, u_tilde, alpha = self.attention_layer(h_1_drop, ctx, ctx_mask)

        if not h_1_res is None:
            # for i in range(10):
            #     print('res',h_1_res[0][i*8:(i+1)*8])
            #     print('aft',h_tilde[0][i*8:(i+1)*8])
            #     print()
            h_tilde = h_tilde + h_1_res 
            

        # Adding Dropout
        h_tilde_drop = self.drop(h_tilde)

        # if not already_dropfeat:
        #     cand_feat[..., :-args.angle_feat_size] = self.drop_env(cand_feat[..., :-args.angle_feat_size])

        if not already_dropfeat and not cand_feat is None:
            f = self.drop_env(cand_feat[..., :-args.angle_feat_size])
            a = cand_feat[..., -args.angle_feat_size:]
            cand_feat = torch.cat([f, a],-1)
        
        if not cand_feat is None:
            _, _, logit = self.candidate_att_layer(h_tilde_drop, cand_feat, output_prob=False)

            if cand_flag:
                seq_len = cand_feat.shape[1]
                h_tilde_drop_repeat = h_tilde_drop.repeat(seq_len,1,1).permute(1,0,2)
                feature_in = torch.cat([cand_feat, h_tilde_drop_repeat], -1) # batch x seq_len x feature_size + dim
                return h_1, c_1, logit, h_tilde, attn_feat, u_tilde, feature_in

            return h_1, c_1, logit, h_tilde, attn_feat, u_tilde
        
        return h_1, c_1, h_tilde, attn_feat, u_tilde


#
class AttnGobalPolicyLSTM_v2(nn.Module):
    ''' An unrolled LSTM with attention over instructions for decoding navigation actions. '''

    def __init__(self, hidden_size,
                       dropout_ratio, feature_size=2048+4):
        super(AttnGobalPolicyLSTM_v2, self).__init__()
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        
        self.drop = nn.Dropout(p=dropout_ratio)
        self.drop_env = nn.Dropout(p=args.featdropout)

        self.candidate_att_layer = SoftDotAttention(hidden_size, feature_size)
        self.linear = nn.Linear(hidden_size, feature_size, bias=False)
        self.linear_entropy = nn.Linear(1, 1, bias=True)
        self.lstm = nn.LSTMCell(feature_size, hidden_size)

    def forward(self, cand_feat, prev_h1, c_0, entropy
                ):
        '''
        Takes a single step in the decoder LSTM (allowing sampling).
        cand_feat: batch x cand x (feature_size + angle_feat_size)
        h1: batch x hidden_size
        '''

        # if not already_dropfeat and not cand_feat is None:
        #     f = self.drop_env(cand_feat[..., :-args.angle_feat_size])
        #     a = cand_feat[..., -args.angle_feat_size:]
        #     cand_feat = torch.cat([f, a],-1)

        # h1_drop = self.drop(h1)

        entropy = entropy.unsqueeze(1) # batch x 1


        _, attn_cand_feat, _ = self.candidate_att_layer(prev_h1, cand_feat)

        h_1, c_1 = self.lstm(attn_cand_feat, (prev_h1, c_0))
        

        attn_cand_feat = attn_cand_feat.unsqueeze(1) # batch x 1 x dim

        target = self.linear(h_1).unsqueeze(2) # batch x dim x 1
        logit = torch.bmm(attn_cand_feat,target).squeeze() + self.linear_entropy(entropy).squeeze() # batch

        prob = torch.sigmoid(logit)
        
        return prob, attn_cand_feat.squeeze(), h_1, c_1


class FullyConnected2(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(FullyConnected2, self).__init__()
        self.lrelu = nn.LeakyReLU(0.1)
        self.linear_layer = nn.Linear(hidden_size, hidden_size, bias=True)
        self.linear_layer_1 = nn.Linear(hidden_size, output_size, bias=False)

    def forward(self, input):
        out = self.lrelu(self.linear_layer(input))
        return self.linear_layer_1(out)

class FullyConnected(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(FullyConnected, self).__init__()
        self.lrelu = nn.LeakyReLU(0.1)
        self.linear_layer = nn.Linear(hidden_size, output_size, bias=False)

    def forward(self, input):
        out = self.lrelu(self.linear_layer(input))
        return out



class Critic(nn.Module):
    def __init__(self,in_dim=args.rnn_dim):
        super(Critic, self).__init__()
        self.state2value = nn.Sequential(
            nn.Linear(in_dim, args.rnn_dim),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(args.rnn_dim, 1),
        )

    def forward(self, state):
        return self.state2value(state).squeeze()

class SpeakerEncoder(nn.Module):
    def __init__(self, feature_size, hidden_size, dropout_ratio, bidirectional):
        super().__init__()
        self.num_directions = 2 if bidirectional else 1
        self.hidden_size = hidden_size
        self.num_layers = 1
        self.feature_size = feature_size

        if bidirectional:
            print("BIDIR in speaker encoder!!")

        self.lstm = nn.LSTM(feature_size, self.hidden_size // self.num_directions, self.num_layers,
                            batch_first=True, dropout=dropout_ratio, bidirectional=bidirectional)
        self.drop = nn.Dropout(p=dropout_ratio)
        self.drop3 = nn.Dropout(p=args.featdropout)
        self.attention_layer = SoftDotAttention(self.hidden_size, feature_size)

        self.post_lstm = nn.LSTM(self.hidden_size, self.hidden_size // self.num_directions, self.num_layers,
                                 batch_first=True, dropout=dropout_ratio, bidirectional=bidirectional)

    def forward(self, action_embeds, feature, lengths, already_dropfeat=False):
        """
        :param action_embeds: (batch_size, length, 2052). The feature of the view
        :param feature: (batch_size, length, 36, 2052). The action taken (with the image feature)
        :param lengths: Not used in it
        :return: context with shape (batch_size, length, hidden_size)
        """
        x = action_embeds
        if not already_dropfeat:
            x[..., :-args.angle_feat_size] = self.drop3(x[..., :-args.angle_feat_size])            # Do not dropout the spatial features

        # LSTM on the action embed
        ctx, _ = self.lstm(x)
        ctx = self.drop(ctx)

        # Att and Handle with the shape
        batch_size, max_length, _ = ctx.size()
        if not already_dropfeat:
            feature[..., :-args.angle_feat_size] = self.drop3(feature[..., :-args.angle_feat_size])   # Dropout the image feature
        x, _, _ = self.attention_layer(                        # Attend to the feature map
            ctx.contiguous().view(-1, self.hidden_size),    # (batch, length, hidden) --> (batch x length, hidden)
            feature.view(batch_size * max_length, -1, self.feature_size),        # (batch, length, # of images, feature_size) --> (batch x length, # of images, feature_size)
        )
        x = x.view(batch_size, max_length, -1)
        x = self.drop(x)

        # Post LSTM layer
        x, _ = self.post_lstm(x)
        x = self.drop(x)

        return x

class SpeakerDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, padding_idx, hidden_size, dropout_ratio):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = torch.nn.Embedding(vocab_size, embedding_size, padding_idx)
        self.lstm = nn.LSTM(embedding_size, hidden_size, batch_first=True)
        self.drop = nn.Dropout(dropout_ratio)
        self.attention_layer = SoftDotAttention(hidden_size, hidden_size)
        self.projection = nn.Linear(hidden_size, vocab_size)
        self.baseline_projection = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout_ratio),
            nn.Linear(128, 1)
        )

    def forward(self, words, ctx, ctx_mask, h0, c0):
        embeds = self.embedding(words)
        embeds = self.drop(embeds)
        x, (h1, c1) = self.lstm(embeds, (h0, c0))

        x = self.drop(x) # batch x seq_length x feature_size

        # Get the size
        batchXlength = words.size(0) * words.size(1)
        multiplier = batchXlength // ctx.size(0)         # By using this, it also supports the beam-search
        # print(words.size(1), multiplier, ctx.size(0))
        feature_size = ctx.size(2)

        # Att and Handle with the shape
        # Reshaping x          <the output> --> (b(word)*l(word), r)
        # Expand the ctx from  (b, a, r)    --> (b(word)*l(word), a, r)
        # Expand the ctx_mask  (b, a)       --> (b(word)*l(word), a)
        # x, _, _ = self.attention_layer(
        #     x.contiguous().view(batchXlength, self.hidden_size),
        #     ctx.unsqueeze(1).expand(-1, multiplier, -1, -1).contiguous().view(batchXlength, -1, self.hidden_size),
        #     mask=ctx_mask.unsqueeze(1).expand(-1, multiplier, -1).contiguous().view(batchXlength, -1)
        # )
        x, _, _ = self.attention_layer(
            x.contiguous().view(batchXlength, self.hidden_size),
            ctx.unsqueeze(1).expand(-1, multiplier, -1, -1).contiguous().view(batchXlength, -1, feature_size),
            mask=ctx_mask.unsqueeze(1).expand(-1, multiplier, -1).contiguous().view(batchXlength, -1)
        )
        x = x.view(words.size(0), words.size(1), self.hidden_size)

        # Output the prediction logit
        x = self.drop(x)
        logit = self.projection(x)

        return logit, h1, c1


class SpeakerDecoder_v2(nn.Module):
    def __init__(self, vocab_size, embedding_size, padding_idx, hidden_size, feature_size, dropout_ratio):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = torch.nn.Embedding(vocab_size, embedding_size, padding_idx)
        self.lstm = nn.LSTM(embedding_size, hidden_size, batch_first=True)
        self.drop = nn.Dropout(dropout_ratio)
        self.attention_layer = SoftDotAttention(hidden_size, feature_size)
        self.projection = nn.Linear(hidden_size, vocab_size)
        self.baseline_projection = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout_ratio),
            nn.Linear(128, 1)
        )

    def forward(self, words, ctx, ctx_mask, h0, c0):
        embeds = self.embedding(words)
        embeds = self.drop(embeds)
        x, (h1, c1) = self.lstm(embeds, (h0, c0))

        x = self.drop(x) # batch x seq_length x feature_size

        # Get the size
        batchXlength = words.size(0) * words.size(1)
        multiplier = batchXlength // ctx.size(0)         # By using this, it also supports the beam-search
        # print(words.size(1), multiplier, ctx.size(0))
        feature_size = ctx.size(2)

        # Att and Handle with the shape
        # Reshaping x          <the output> --> (b(word)*l(word), r)
        # Expand the ctx from  (b, a, r)    --> (b(word)*l(word), a, r)
        # Expand the ctx_mask  (b, a)       --> (b(word)*l(word), a)
        # x, _, _ = self.attention_layer(
        #     x.contiguous().view(batchXlength, self.hidden_size),
        #     ctx.unsqueeze(1).expand(-1, multiplier, -1, -1).contiguous().view(batchXlength, -1, self.hidden_size),
        #     mask=ctx_mask.unsqueeze(1).expand(-1, multiplier, -1).contiguous().view(batchXlength, -1)
        # )
        x, _, _ = self.attention_layer(
            x.contiguous().view(batchXlength, self.hidden_size),
            ctx.unsqueeze(1).expand(-1, multiplier, -1, -1).contiguous().view(batchXlength, -1, feature_size),
            mask=ctx_mask.unsqueeze(1).expand(-1, multiplier, -1).contiguous().view(batchXlength, -1)
        )
        x = x.view(words.size(0), words.size(1), self.hidden_size)

        # Output the prediction logit
        x = self.drop(x)
        logit = self.projection(x)

        return logit, h1, c1


