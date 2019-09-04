import torch.nn as nn


class EmbeddingLayer(nn.Module):
    def __init__(self, feature_dim, embedding_dim):
        super().__init__()

        self.embed = nn.Embedding(feature_dim, embedding_dim, padding_idx=0)
        
        # normal weight initialization
        self.embed.weight.data.normal_(0., 0.0001)
        # TODO: regularization

    def forward(self, x):
        return self.embed(x)



if __name__ == "__main__":
    a = EmbeddingLayer(10, 12)
    import torch
    b = torch.ones((2048,)).type(torch.LongTensor)
    print(a(b).size())