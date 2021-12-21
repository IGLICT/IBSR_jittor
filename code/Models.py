import jittor as jt
from jittor import nn, models
if jt.has_cuda:
    jt.flags.use_cuda = 1 # jt.flags.use_cuda

class QueryEncoder(nn.Module):
    def __init__(self, out_dim=128):
        super(QueryEncoder, self).__init__()
        self.dim = out_dim
        self.resnet = models.resnet50(pretrained=False)
        self.resnet.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        fc_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
                nn.BatchNorm1d(fc_features*1),
                nn.Linear(fc_features*1, self.dim),
            )

    def execute(self, input):
        embeddings = self.resnet(input)
        embeddings = jt.normalize(embeddings, p=2, dim=1)
        return embeddings


class RenderingEncoder(nn.Module):
    def __init__(self, out_dim=128):
        super(RenderingEncoder, self).__init__()
        self.dim = out_dim
        self.resnet = models.resnet18(pretrained=False)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        fc_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
                nn.BatchNorm1d(fc_features*1),
                nn.Linear(fc_features*1, self.dim),
            )

    def execute(self, inputs):
        embeddings = self.resnet(inputs)
        embeddings = jt.normalize(embeddings, p=2, dim=1)
        return embeddings


class Attention(nn.Module):
    '''
    Revised from pytorch version: <https://github.com/IBM/pytorch-seq2seq/blob/master/LICENSE> 
    '''

    """ Applies attention mechanism on the `context` using the `query`.

    **Thank you** to IBM for their initial implementation of :class:`Attention`. Here is
    their `License
    <https://github.com/IBM/pytorch-seq2seq/blob/master/LICENSE>`__.

    Args:
        dimensions (int): Dimensionality of the query and context.
        attention_type (str, optional): How to compute the attention score:

            * dot: :math:`score(H_j,q) = H_j^T q`
            * general: :math:`score(H_j, q) = H_j^T W_a q`

    Example:

         >>> attention = Attention(256)
         >>> query = torch.randn(5, 1, 256)
         >>> context = torch.randn(5, 5, 256)
         >>> output, weights = attention(query, context)
         >>> output.size()
         torch.Size([5, 1, 256])
         >>> weights.size()
         torch.Size([5, 1, 5])
    """

    def __init__(self, dimensions, attention_type='general'):
        super(Attention, self).__init__()

        if attention_type not in ['dot', 'general']:
            raise ValueError('Invalid attention type selected.')

        self.attention_type = attention_type
        if self.attention_type == 'general':
            self.linear_in = nn.Linear(dimensions, dimensions, bias=False)

        self.linear_out = nn.Linear(dimensions * 2, dimensions, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()

    def execute(self, query, context):
        """
        Args:
            query (:class:`torch.FloatTensor` [batch size, output length, dimensions]): Sequence of
                queries to query the context.
            context (:class:`torch.FloatTensor` [batch size, query length, dimensions]): Data
                overwhich to apply the attention mechanism.

        Returns:
            :class:`tuple` with `output` and `weights`:
            * **output** (:class:`torch.LongTensor` [batch size, output length, dimensions]):
              Tensor containing the attended features.
            * **weights** (:class:`torch.FloatTensor` [batch size, output length, query length]):
              Tensor containing attention weights.
        """
        batch_size, output_len, dimensions = query.size()
        query_len = context.size(1)

        if self.attention_type == "general":
            query = query.view(batch_size * output_len, dimensions)
            query = self.linear_in(query)
            query = query.view(batch_size, output_len, dimensions)

        # TODO: Include mask on PADDING_INDEX?

        # (batch_size, output_len, dimensions) * (batch_size, query_len, dimensions) ->
        # (batch_size, output_len, query_len)
        # attention_scores = nn.bmm(query, context.transpose(1, 2).contiguous())
        attention_scores = nn.bmm(query, context.transpose(0, 2, 1))

        # Compute weights across every context sequence
        attention_scores = attention_scores.view(batch_size * output_len, query_len)
        attention_weights = self.softmax(attention_scores)
        attention_weights = attention_weights.view(batch_size, output_len, query_len)

        # (batch_size, output_len, query_len) * (batch_size, query_len, dimensions) ->
        # (batch_size, output_len, dimensions)
        mix = nn.bmm(attention_weights, context)
 
        # concat -> (batch_size * output_len, 2*dimensions)
        combined = jt.concat((mix, query), dim=2)
        combined = combined.view(batch_size * output_len, 2 * dimensions)

        # Apply linear_out on every 2nd dimension of concat
        # output -> (batch_size, output_len, dimensions)
        output = self.linear_out(combined).view(batch_size, output_len, dimensions)
        output = self.tanh(output)

        return output, attention_weights


class RetrievalNet(nn.Module):
    '''
    QueryEncoder
    RenderingEncoder
    Attention
    '''
    def __init__(self, cfg):
        super(RetrievalNet, self).__init__()
        self.dim = cfg.models.z_dim
        self.size = cfg.data.pix_size
        self.view_num = cfg.data.view_num
        self.query_encoder = QueryEncoder(self.dim)
        self.rendering_encoder = RenderingEncoder(self.dim)
        self.attention = Attention(self.dim)

       
    def execute(self, query, rendering):
        query_ebd = self.get_query_ebd(query)
        bs = query_ebd.shape[0]
        rendering = rendering.view(-1, 1, self.size, self.size)
        rendering_ebds = self.get_rendering_ebd(rendering).view(-1, self.view_num, self.dim)

        #(shape, image, ebd) -> (bs, bs, 128)
        query_ebd = query_ebd.unsqueeze(0).repeat(bs, 1, 1)
        # query_ebd:    bs, bs, dim
        # rendering_ebds:  bs, 12, dim
        _, weights = self.attention_query(query_ebd, rendering_ebds)

        # weights:                  bxxbsx12
        # rendering_ebds:           bsx12x128
        # queried_rendering_ebd:    bsxbsx128   (shape, model, 128)
        # reference to https://pytorchnlp.readthedocs.io/en/latest/_modules/torchnlp/nn/attention.html#Attentionl
        queried_rendering_ebd = nn.bmm(weights, rendering_ebds)
        return query_ebd, queried_rendering_ebd       

    def get_query_ebd(self, inputs):
        return self.query_encoder(inputs)
    
    def get_rendering_ebd(self, inputs):
        return self.rendering_encoder(inputs)

    def attention_query(self, ebd, pool_ebd):
        return self.attention(ebd, pool_ebd)



if __name__ == '__main__':
    import yaml
    import argparse

    with open('./configs/pix3d.yaml', 'r') as f:
        config = yaml.load(f)
    def dict2namespace(config):
        namespace = argparse.Namespace()
        for key, value in config.items():
            if isinstance(value, dict):
                new_value = dict2namespace(value)
            else:
                new_value = value
            setattr(namespace, key, new_value)
        return namespace
    config = dict2namespace(config)




    models = RetrievalNet(config)
    img = jt.random([2,4,224,224]).stop_grad()
    mask = jt.random([2,12,224,224]).stop_grad()

    # mm = models.resnet50(pretrained=False)
    # # print(mm)
    # a = mm(img) 

    outputs = models(img, mask)