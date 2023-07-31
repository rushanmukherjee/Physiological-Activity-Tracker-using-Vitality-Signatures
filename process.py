from cmath import nan



import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from sklearn import preprocessing



df_m = pd.read_csv("Acceleration with g baseline 2022-06-27 2-31-10 PM.csv")

df_2 = pd.read_csv('Custom Data - 2022-06-27 - Recording_Maqsood_Baseline.csv')
df_2 = df_2[df_2.SPO != -999]
df_2 = df_2[df_2.Heart_Rate!=-999]
df_2 = df_2[:300].reset_index(drop=True)
df_m = df_m.drop(columns=['Absolute acceleration'])
df_m = df_m[:300].reset_index(drop=True)
df_m= pd.concat([df_m,df_2],axis=1)


x = df_m.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df_m = pd.DataFrame(x_scaled)
df_m_s = pd.read_csv("Acceleration with g steps 2022-06-27 2-26-25 PM.csv")
df = pd.read_csv('Custom Data - 2022-06-27 - Recording_Maqsood_steps.csv')
df = df[df.SPO != -999]
df = df[df.Heart_Rate!=-999]
df = df[:300].reset_index(drop=True)
df_m_s = df_m_s.drop(columns=['Absolute acceleration'])
df_m_s = df_m_s[:300].reset_index(drop=True)
df_m_s= pd.concat([df_m_s,df],axis=1)


x = df_m_s.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df_m_s = pd.DataFrame(x_scaled)


###emulate
heart_em = pd.DataFrame(np.random.normal(120, 32, 300))


df_m_p = pd.read_csv("Acceleration with g gowtham planks 2022-06-27 3-58-17 PM.csv")
df_p = pd.read_csv('Custom Data - 2022-06-27 - Recording_Gowtham_Plank.csv')

df_p = df_p[df_p.SPO != -999]
df_p = df_p[df_p.Heart_Rate!=-999]
df_p = df_p[:300].reset_index(drop=True)
# df_p['Heart_Rate'] = heart_em
df_m_p = df_m_p.drop(columns=['Absolute acceleration'])
df_m_p = df_m_p[:300].reset_index(drop=True)
df_m_p= pd.concat([df_m_p,df_p],axis=1)


x = df_m_p.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df_m_p = pd.DataFrame(x_scaled)

df_m = pd.concat([df_m,df_m_s,df_m_p])
# print(dfs_m)
lab = ['baseline']*300+['steps']*300+['planks']*300

#training

from sklearn.decomposition import PCA
def appy_pca(df_m):
    pca = PCA(n_components=3)
    principalComponents = pca.fit_transform(df_m)
    principalDf = pd.DataFrame(data = principalComponents
                , columns = ['principal component 1', 'principal component 2','principal component 3'])

    # principalDf['dataset']  =lab
    return principalDf
from sklearn.mixture import GaussianMixture
gm = GaussianMixture(n_components=3, covariance_type='diag',random_state=42)
principalDf = appy_pca(df_m)
principalDf['dataset']  =lab
# ##plotly
# import plotly.express as px
# fig = px.scatter_3d( principalDf, x = "principal component 1", y = "principal component 2",z = 'principal component 3',color = 'dataset')
# # fig.show()
df_m=appy_pca(df_m)
gm.fit(df_m)
labels_val = [0]*300+[1]*300+[2]*300
print(f'training accuracy:{(labels_val==gm.predict(df_m)).sum()*100/900}')
print(gm.converged_)
print(gm.n_iter_)







#Functions to visualize data
import matplotlib.cm as cmx
import numpy as np
def plot_sphere(w=0, c=[0,0,0], r=[1, 1, 1], subdev=10, ax=None, sigma_multiplier=3):
    '''
        plot a sphere surface
        Input: 
            c: 3 elements list, sphere center
            r: 3 element list, sphere original scale in each axis ( allowing to draw elipsoids)
            subdiv: scalar, number of subdivisions (subdivision^2 points sampled on the surface)
            ax: optional pyplot axis object to plot the sphere in.
            sigma_multiplier: sphere additional scale (choosing an std value when plotting gaussians)
        Output:
            ax: pyplot axis object
    '''

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    pi = np.pi
    cos = np.cos
    sin = np.sin
    phi, theta = np.mgrid[0.0:pi:complex(0,subdev), 0.0:2.0 * pi:complex(0,subdev)]
    x = sigma_multiplier*r[0] * sin(phi) * cos(theta) + c[0]
    y = sigma_multiplier*r[1] * sin(phi) * sin(theta) + c[1]
    z = sigma_multiplier*r[2] * cos(phi) + c[2]
    cmap = cmx.ScalarMappable()
    cmap.set_cmap('jet')
    c = cmap.to_rgba(w)

    ax.plot_surface(x, y, z, color=c, alpha=0.2, linewidth=1)
    # u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    # x = sigma_multiplier*r[0] *np.cos(u)*np.sin(v)+ c[0]
    # y = sigma_multiplier*r[1] *np.sin(u)*np.sin(v)+ c[1]
    # z = sigma_multiplier*r[2] *np.cos(v)+ c[1]
    # ax.plot_surface(x,y,z,color ='r' )
    return ax

def visualize_3d_gmm(points, w, mu, stdev):
    '''
    plots points and their corresponding gmm model in 3D
    Input: 
        points: N X 3, sampled points
        w: n_gaussians, gmm weights
        mu: 3 X n_gaussians, gmm means
        stdev: 3 X n_gaussians, gmm standard deviation (assuming diagonal covariance matrix)
    Output:
        None
    '''

    n_gaussians = mu.shape[1]
    N = int(np.round(points.shape[0] / n_gaussians))
    # Visualize data
    fig = plt.figure(figsize=(8, 8))
    axes = fig.add_subplot(111, projection='3d')
    axes.set_xlim([-1, 1])
    axes.set_ylim([-1, 1])
    axes.set_zlim([-1, 1])
    plt.set_cmap('Set1')
    colors = cmx.Set1(np.linspace(0, 1, n_gaussians))
    points = np.array(points)
    # axes.scatter(points[:, 0], points[:, 1], points[:, 2], alpha=0.3)
    for i in range(n_gaussians):
        idx = range(i * N, (i + 1) * N)
        axes.scatter(points[idx, 0], points[idx, 1], points[idx, 2], alpha=0.3, c=colors[i])
        plot_sphere(w=w[i], c=mu[:, i], r=stdev[:, i], ax=axes)

    plt.title('3D GMM')
    axes.set_xlabel('X')
    axes.set_ylabel('Y')
    axes.set_zlabel('Z')
    axes.view_init(35.246, 45)
    plt.show()

visualize_3d_gmm( df_m, gm.weights_.reshape(-1,1), gm.means_.T, np.sqrt(abs(gm.covariances_)).T)

# sns.scatterplot(data = principalDf, x = "principal component 1", y = "principal component 2",style = 'dataset')
