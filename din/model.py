import torch.nn as nn
import torch

from .embedding import EmbeddingLayer
from .fc import FullyConnectedLayer
from .attention import AttentionSequencePoolingLayer



dim_config = {
    'user_exposed_time': 24,
    'user_gender': 2,
    'user_age': 9,
    'history_article_id': 53932,   # multi-hot
    'history_image_feature': 2048,
    'history_categories': 23,
    'query_article_id': 1856,    # one-hot
    'query_image_feature': 2048,
    'query_categories': 23
}

que_embed_features = ['query_article_id']
que_image_features = ['query_image_feature']
que_category =  ['query_categories']

his_embed_features = ['history_article_id']
his_image_features = ['history_image_feature']
his_category =  ['history_categories']

image_hidden_dim = 64
category_dim = 23

embed_features = [k for k, _ in dim_config.items() if 'user' in k]


class DeepInterestNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        embedding_size = config['embedding_size']

        self.query_feature_embedding_dict = dict()
        for feature in que_embed_features:
            self.query_feature_embedding_dict[feature] = EmbeddingLayer(feature_dim=dim_config[feature],
                                                                        embedding_dim=embedding_size).cuda()
        self.query_image_fc = FullyConnectedLayer(input_size=2048,
                                                  hidden_size=[image_hidden_dim],
                                                  bias=[True],
                                                  activation='relu').cuda()
        
        self.history_feature_embedding_dict = dict()
        for feature in his_embed_features:
            self.history_feature_embedding_dict[feature] = EmbeddingLayer(feature_dim=dim_config[feature],
                                                                          embedding_dim=embedding_size).cuda()     
        self.history_image_fc = FullyConnectedLayer(input_size=2048,
                                                    hidden_size=[image_hidden_dim],
                                                    bias=[True],
                                                    activation='relu').cuda()                                                      

        self.attn = AttentionSequencePoolingLayer(embedding_dim=image_hidden_dim + embedding_size + category_dim).cuda()
        self.fc_layer = FullyConnectedLayer(input_size=2 * (image_hidden_dim + embedding_size + category_dim) + sum([dim_config[k] for k in embed_features]),
                                            hidden_size=[200, 80, 1],
                                            bias=[True, True, False],
                                            activation='relu',
                                            sigmoid=True).cuda()

    def forward(self, user_features):
        # user_features -> dict (key:feature name, value: feature tensor)

        # deep input embedding
        feature_embedded = []

        for feature in embed_features:
            feature_embedded.append(user_features[feature])

        feature_embedded = torch.cat(feature_embedded, dim=1)
        #print('User_feature_embed size', user_feature_embedded.size()) # batch_size * (feature_size * embedding_size)
        #print('User feature done')

        query_feature_embedded = []

        for feature in que_embed_features:
            query_feature_embedded.append(self.query_feature_embedding_dict[feature](user_features[feature].squeeze()))
        for feature in que_image_features:
            query_feature_embedded.append(self.query_image_fc(user_features[feature]))
        for feature in que_category:
            query_feature_embedded.append(user_features[feature])

        query_feature_embedded = torch.cat(query_feature_embedded, dim=1)
        # print('Query feature_embed size', query_feature_embedded.size()) # batch_size * (feature_size * embedding_size)
        # print('Query feature done')
        # exit()

        # TODO: history
        history_feature_embedded = []
        for feature in his_embed_features:
            #print(feature)
            #print(user_features[feature].size())
            history_feature_embedded.append(self.history_feature_embedding_dict[feature](user_features[feature]))
            #print(self.history_feature_embedding_dict[feature](user_features[feature]).size())

        for feature in his_image_features:
            #print(user_features[feature].size())
            history_feature_embedded.append(self.history_image_fc(user_features[feature]))
        for feature in his_category:
            history_feature_embedded.append(user_features[feature])

        history_feature_embedded = torch.cat(history_feature_embedded, dim=2)
        #print('History feature_embed size', history_feature_embedded.size()) # batch_size * T * (feature_size * embedding_size)
        #print('History feature done')
        
        #print(user_features['history_len'])
        #print(user_features['history_len'].size())
        
        
        history = self.attn(query_feature_embedded.unsqueeze(1), 
                            history_feature_embedded, 
                            user_features['history_len']) 
        
        concat_feature = torch.cat([feature_embedded, query_feature_embedded, history.squeeze()], dim=1)
        
        # fully-connected layers
        #print(concat_feature.size())
        output = self.fc_layer(concat_feature)
        return output


if __name__ == "__main__":
    a = DeepInterestNetwork()
    import torch
    import numpy as np

    
    user_feature = {
        'user_exposed_time': torch.LongTensor(np.zeros(shape=(2, 24))),
        'user_gender': torch.LongTensor(np.zeros(shape=(2, 2))),
        'user_age': torch.LongTensor(np.zeros(shape=(2, 9))),
    }
    a(user_feature)