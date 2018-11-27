import os
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from matplotlib import cm

def tSNE_Visu(feat_t0,feat_t1,label):

    n, c, h, w = feat_t0.data.shape
    feat_t0 = torch.transpose(feat_t0.view(c, h * w), 1, 0)
    feat_t1 = torch.transpose(feat_t1.view(c, h * w), 1, 0)
    labels = torch.transpose(label.view(1, h * w), 1, 0)
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
    plot_dim = 300
    low_dim_embs_t0 = tsne.fit_transform(feat_t0.data.cpu().numpy())[:plot_dim]
    low_dim_embs_t1 = tsne.fit_transform(feat_t1.data.cpu().numpy())[:plot_dim]
    labels = labels.numpy()[:plot_dim]
    #labels = test_y.numpy()[:plot_only]
    plot_with_labels([low_dim_embs_t0,low_dim_embs_t1], labels,h * w)

def plot_with_labels(lowDWeights, labels,sz):
    plt.cla()
    X_t0,Y_t0 = lowDWeights[0][:,0],lowDWeights[0][:,1]
    X_t1,Y_t1 = lowDWeights[1][:,0],lowDWeights[1][:,1]
    for idx,(x_t0,y_t0,x_t1,y_t1,lab) in enumerate(zip(X_t0,Y_t0,X_t1,Y_t1,labels)):
        c = cm.rainbow(int(255 * idx/sz))
        plt.text(x_t0,y_t0,lab,backgroundcolor=c,fontsize=9)
        plt.text(x_t1,y_t1,lab,backgroundcolor=c,fontsize=9)
    plt.xlim(X_t0.min(), X_t0.max());plt.ylim(Y_t0.min(), Y_t1.max());
    plt.title('Visualize last layer');plt.show();plt.pause(0.01)
        #for x, y, s in zip(X, Y, labels):
        #c = cm.rainbow(int(255 * s / 9)); plt.text(x, y, s, backgroundcolor=c, fontsize=9)

def tSNE_Visu_feature_cat(feat,label,save_dir,title='Visualize last layer'):

    n, c, h, w = feat.data.shape
    feat = torch.transpose(feat.view(c, h * w), 1, 0)
    labels = torch.transpose(label.view(1, h * w), 1, 0)
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=500)
    low_dim_embs = tsne.fit_transform(feat.data.cpu().numpy())
    labels = labels.numpy()
    plot_with_labels_feat_cat(low_dim_embs, labels,save_dir,title)
    #plot_with_labels_feat_cat_without_text(low_dim_embs,labels,save_dir)

def plot_with_labels_feat_cat(lowDWeights, labels,save_dir,title):
    plt.cla()
    X,Y = lowDWeights[:,0],lowDWeights[:,1]
    #plt.scatter(X,Y)
    for idx,(x,y,lab) in enumerate(zip(X,Y,labels)):
        color = cm.rainbow(int(255 * lab/2))
        #plt.scatter(x,y,color)
        plt.text(x,y,lab,backgroundcolor=color,fontsize=0)
    plt.xlim(X.min() *2 , X.max() *2);plt.ylim(Y.min()*2, Y.max()*2)
    plt.title(title)
    #plt.show();plt.pause(0.01)
    plt.savefig(save_dir)
    print save_dir
    #for x, y, s in zip(X, Y, labels):
    #c = cm.rainbow(int(255 * s / 9)); plt.text(x, y, s, backgroundcolor=c, fontsize=9)

def plot_with_labels_feat_cat_without_text(lowDWeights, labels,save_dir):
    plt.cla()
    X,Y = lowDWeights[:,0],lowDWeights[:,1]
    for idx,(x,y,lab) in enumerate(zip(X,Y,labels)):
        #c = cm.rainbow(int(255 * lab/2))
        if lab == 0:
           plt.plot(x,y,'b')
        if lab == 1:
           plt.plot(x,y,'r')
        #plt.text(x,y,lab,backgroundcolor=c,fontsize=9)
    plt.xlim(X.min() *2 , X.max() *2);plt.ylim(Y.min()*2, Y.max()*2)
    plt.title('Visualize last layer')
    #plt.show();plt.pause(0.01)
    plt.savefig(save_dir)
    print save_dir
