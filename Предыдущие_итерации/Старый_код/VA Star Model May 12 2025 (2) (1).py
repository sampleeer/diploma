#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import scipy as sp
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.optimize as opt
import ot  # Для транспортной метрики
from scipy.stats import uniform, truncnorm
from scipy.optimize import differential_evolution


# In[5]:


#pip install pot
get_ipython().system('pip install  PyQt5')


# In[3]:


get_ipython().system('pip install PySide2')


# In[48]:


get_ipython().run_line_magic('matplotlib', 'qt')


# In[49]:


def plot_3d(C):
    # Create figure and 3D axis
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    ax.set_zlim(22, 0)

    # Plot the voxels where value == 1
    ax.voxels(C, edgecolor='k')


# In[50]:


def compare_XY(X, Y):
    # Create figure and 3D axis
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    ax.set_zlim(22, 0)

    # Plot the voxels where value == 1
    ax.voxels(X, edgecolor='k')
    ax.voxels(Y, edgecolor='r')


# In[51]:


def compare_proj(X, Y):
    # Create figure and 3D axis
    fig, ax = plt.subplots(1, 2)
    
    ax[0].matshow(X.any(axis=0).transpose() + 5 * Y.any(axis=0).transpose() , cmap='Greys')
    ax[1].matshow(X.any(axis=1).transpose() + 5 * Y.any(axis=1).transpose() , cmap='Greys')
    
    ax[0].set_aspect(96 / 22)
    ax[0].set_xlim(0, 96)
    ax[0].set_ylim(22, 0)
    ax[1].set_aspect(96 / 22)
    ax[1].set_xlim(0, 96)
    ax[1].set_ylim(22, 0)


# In[52]:


def x_proj(C):
    return C.any(axis=0)

def y_proj(C):
    return C.any(axis=1)


# In[53]:


def plot_X_projection(C):
    plt.matshow(C.any(axis=0).transpose(), cmap='Greys')

def plot_Y_projection(C):
    plt.matshow(C.any(axis=1).transpose(), cmap='Greys')
    
def plot_projections(C):
    fig, ax = plt.subplots(1, 2)
    
    ax[0].matshow(C.any(axis=0).transpose(), cmap='Greys')
    ax[1].matshow(C.any(axis=1).transpose(), cmap='Greys')
    
    ax[0].set_aspect(96 / 22)
    ax[0].set_xlim(0, 96)
    ax[0].set_ylim(22, 0)
    ax[1].set_aspect(96 / 22)
    ax[1].set_xlim(0, 96)
    ax[1].set_ylim(22, 0)


# In[54]:


def check_XY_bounds(x, xmin=0, xmax=95):
    return (x >= xmin) & (x <= xmax)


# ### Опишем модель взаимодействия

# In[55]:


# # Опишем модель "звезды", т.е. у нас есть набор параметров:
# # точка и углы влёта, точка взаимодействия и направления разлёта частиц.
# # Мы подгоняем параметры этой модели под данные.

# startx, starty, startz = 34.0, 45.0, 0.0
# theta = np.pi / 4
# phi = np.pi / 3

# line = (startx, starty, startz, theta, phi)  # Набор, определяющий прямую
# l = (np.tan(theta) * np.cos(phi), np.tan(theta) * np.sin(phi), 1.0)  # Направляющий вектор прямой с координатой z = 1


# In[56]:


# # Это у нас одна прямая
# lx = startx + l[0] * np.arange(0, 22, 1)
# ly = starty + l[1] * np.arange(0, 22, 1)
# lz = np.arange(0, 22, 1)


# In[57]:


# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# ax.plot(lx, ly, lz, '-ok')
# ax.set_zlim(np.max(lz), np.min(lz))


# In[58]:


# # Теперь надо описать модель звезды

# zint = 7  # Точка взаимодействия
# npart = 3   # Количество порожденных частиц (возможно, эту величину стоит сделать стохастической)
# maxz = 22  # Глубина калориметра
# thetapart = np.array([0, 2 * np.pi / 3, np.pi / 3])
# phipart = np.array([0.0, np.pi / 6, -np.pi / 4])
# direction = thetapart < np.pi / 2  # Вверх или вниз летит частица


# In[59]:


# line = (startx, starty, startz, theta, phi)  # Набор, определяющий прямую
# l = (np.tan(theta) * np.cos(phi), np.tan(theta) * np.sin(phi), 1.0)  # Направляющий вектор прямой с координатой z = 1

# # Это у нас одна прямая
# lx = startx + l[0] * np.arange(0, zint + 1, 1)
# ly = starty + l[1] * np.arange(0, zint + 1, 1)
# lz = np.arange(0, zint + 1, 1)


# In[60]:


# # Координаты X, Y точки взаимодействия
# xint, yint = lx[zint], ly[zint]


# In[61]:


# lines = dict()

# for line_num in range(npart):
#     if direction[line_num]:
#         steps = maxz - zint + 1
#         lines[line_num] = [
#             xint + np.tan(thetapart[line_num]) * np.cos(phipart[line_num]) * np.arange(0, steps - 1, 1),
#             yint + np.tan(thetapart[line_num]) * np.cos(phipart[line_num]) * np.arange(0, steps - 1, 1),
#             np.arange(zint, maxz, 1)
#         ]
#     else:
#         steps = zint + 1
#         lines[line_num] = [
#             xint - np.tan(thetapart[line_num]) * np.cos(phipart[line_num]) * np.arange(0, steps, 1),
#             yint - np.tan(thetapart[line_num]) * np.cos(phipart[line_num]) * np.arange(0, steps, 1),
#             np.arange(zint, -1, -1)
#         ]


# In[62]:


# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# ax.plot(lx, ly, lz, '-ok')

# for line_num in range(npart):
#     ax.plot(*lines[line_num], '-or')
    
# ax.plot([xint], [yint], [zint], 'ok', ms=20)


# ax.set_zlim(22, 0)
# ax.set_xlim(-11, 11)
# ax.set_ylim(-11, 11)


# ### Процедуры генерации событий

# In[63]:


# StartX 0..95, StartY 0..95, start_theta 0..?, start_phi -pi..pi
def _generate_event(startx, starty, start_theta, start_phi, zint, npart, theta_part, phi_part):
    startz = 0
    maxz = 21
    l = (np.tan(start_theta) * np.cos(start_phi), np.tan(start_theta) * np.sin(start_phi), 1)  # Направляющий вектор прямой с координатой z = 1
    lx = startx + l[0] * np.arange(0, zint + 1, 1)
    ly = starty + l[1] * np.arange(0, zint + 1, 1)
    lz = np.arange(0, zint + 1, 1, dtype=int)
    zint = round(zint)
    xint, yint = lx[zint], ly[zint]  # Координаты X, Y точки взаимодействия
    
    C = np.zeros((96, 96, 22), dtype=int)
    
    # Флаг прерванного трека (вылет за пределы калориметра) 
    track_interrupted = False
    
    # Проверка того, вышли ли мы за пределы калориметра
    # Если вылетели, то обрываем траекторию
    if not(check_XY_bounds(xint) and check_XY_bounds(yint)): 
        idx = check_XY_bounds(lx) & check_XY_bounds(ly)
        lx, ly, lz = lx[idx], ly[idx], lz[idx]
        track_interrupted = True
        
#     print(lx, ly)
#     print(xint, yint, zint)
    
    lx_int = np.round(lx).astype(int)
    ly_int = np.round(ly).astype(int)
    
    C[lx_int, ly_int, lz] = 1  # Добавили траекторию первичной частицы
    
    
    if not(track_interrupted):
        lines = dict()
        
        direction = np.array(theta_part) < np.pi / 2

        for line_num in range(npart):
            newline = []
            if direction[line_num]:
                steps = maxz - zint + 1
                newline = [
                    xint + np.tan(theta_part[line_num]) * np.cos(phi_part[line_num]) * np.arange(0, steps - 1, 1),
                    yint + np.tan(theta_part[line_num]) * np.cos(phi_part[line_num]) * np.arange(0, steps - 1, 1),
                    np.arange(zint, maxz, 1, dtype=int)
                ]
            else:
                steps = zint + 1
                newline = [
                    xint - np.tan(theta_part[line_num]) * np.cos(phi_part[line_num]) * np.arange(0, steps, 1),
                    yint - np.tan(theta_part[line_num]) * np.cos(phi_part[line_num]) * np.arange(0, steps, 1),
                    np.arange(zint, -1, -1, dtype=int)
                ]
            # Отрезаем значения вне прибора
            idx = (newline[0] >= 0) & (newline[0] <= 95) & (newline[1] >= 0) & (newline[1] <= 95)
            lines[line_num] = [newline[0][idx], newline[1][idx], newline[2][idx]]



        for line_num in range(npart):
            C[np.round(lines[line_num][0]).astype(int),
              np.round(lines[line_num][1]).astype(int),
              lines[line_num][2]] = 1
    
    return C


# In[64]:


def _generate_zero_event(startx, starty, theta, phi):
    return _generate_event(startx, starty, theta, phi, 21, 0, [], [])

def _generate_one_event(startx, starty, theta, phi, zint, theta1, phi1):
    return _generate_event(startx, starty, theta, phi, zint, 1, [theta1], [phi1])

def _generate_two_event(startx, starty, theta, phi, zint, theta1, phi1, theta2, phi2):
    return _generate_event(startx, starty, theta, phi, zint, 2, [theta1, theta2], [phi1, phi2])

def _generate_three_event(startx, starty, theta, phi, zint, theta1, phi1, theta2, phi2, theta3, phi3):
    return _generate_event(startx, starty, theta, phi, zint, 3, [theta1, theta2, theta3], [phi1, phi2, phi3])

def _generate_four_event(startx, starty, theta, phi, zint, theta1, phi1, theta2, phi2, theta3, phi3, theta4, phi4):
    return _generate_event(
        startx, starty, theta, phi, zint,
        4,
        [theta1, theta2, theta3, theta4],
        [phi1, phi2, phi3, phi4]
    )


# In[65]:


def plot_3d_event(startx, starty, start_theta, start_phi, zint, npart, theta_part, phi_part):
    X = _generate_event(startx, starty, start_theta, start_phi, zint, npart, theta_part, phi_part)
    plot_3d(X)


# In[66]:


X = _generate_event(34.0, 45.0, np.pi / 4, np.pi / 3, 7, 3, [0, 2 * np.pi / 3, np.pi / 3], [0.0, np.pi / 6, -np.pi / 4])
Y = _generate_event(34.0, 52.0, 0, 0, 21, 0, [], [])
Z = _generate_event(52.0, 34.0, 0, 0, 21, 0, [], [])
A = _generate_event(52.0, 34.0, 0, 0, 6, 1, [np.pi/6], [np.pi/6])
B = _generate_event(46.0, 46.0, np.pi / 2 * 0.9, 0, 21, 0, [], []) 


# In[67]:


# Сделаем выборку тестовых событий
size = 100

test_0_params = dict()
test_0_events = dict()

test_1_params = dict()
test_1_events = dict()

test_2_params = dict()
test_2_events = dict()

test_3_params = dict()
test_3_events = dict()

# Событий с точкой взаимодействия в середине должно быть больше
def generate_startx(size=100):
    loc, scale = 47.5, 20.0
    # Define bounds (in standard normal space)
    lower, upper = -loc / scale, loc / scale
    samples = truncnorm.rvs(lower, upper, loc=loc, scale=scale, size=size)
    integers = np.round(samples)
    return integers

# Событий с точкой взаимодействия в середине должно быть больше
def generate_zint(size=100):
    # Define bounds (in standard normal space)
    lower, upper = (0 - 10.5) / 4.0, (21 - 10.5) / 4.0
    samples = truncnorm.rvs(lower, upper, loc=10.5, scale=4.0, size=size)
    integers = np.round(samples).astype(int)
    return integers

# Азимутальный угол распределён равномерно
def generate_phi_angle(size=100):
    samples = uniform.rvs(0, np.pi * 2, size=size)
    return samples

# Зенитный угол распределён равномерно
def generate_theta_start_angle(size=100):
    scale = 0.3
    # Define bounds (in standard normal space)
    lower, upper = -np.pi / 3 / scale, np.pi / 3 / scale
    samples = truncnorm.rvs(lower, upper, loc=0, scale=scale, size=size)
    return np.abs(samples)

# Зенитный угол распределён равномерно
def generate_theta_int_angle(size=100):
    samples = uniform.rvs(-np.pi / 2, np.pi / 2, size=size)
    return samples


# Zero events (no interaction)
startx = generate_startx(size=size)
starty = generate_startx(size=size)
theta = generate_theta_start_angle(size=size)
phi = generate_phi_angle(size=size)

test_0_params = [(startx[i], starty[i], theta[i], phi[i]) for i in range(size)]
test_0_events = [_generate_zero_event(*test_0_params[i]) for i in range(size)] 
    
# One particle events
startx = generate_startx(size=size)
starty = generate_startx(size=size)
theta = generate_theta_start_angle(size=size)
phi = generate_phi_angle(size=size)
zint = generate_zint(size=size)
theta1 = generate_theta_int_angle(size=size)
phi1 = generate_phi_angle(size=size)

test_1_params = [(startx[i], starty[i], theta[i], phi[i], zint[i], theta1[i], phi1[i]) for i in range(size)]
test_1_events = [_generate_one_event(*test_1_params[i]) for i in range(size)] 

# Two particle events
startx = generate_startx(size=size)
starty = generate_startx(size=size)
theta = generate_theta_start_angle(size=size)
phi = generate_phi_angle(size=size)
zint = generate_zint(size=size)
theta1 = generate_theta_int_angle(size=size)
phi1 = generate_phi_angle(size=size)
theta2 = generate_theta_int_angle(size=size)
phi2 = generate_phi_angle(size=size)

test_2_params = [(startx[i], starty[i], theta[i], phi[i], zint[i], theta1[i], phi1[i], theta2[i], phi2[i]) for i in range(size)]
test_2_events = [_generate_two_event(*test_2_params[i]) for i in range(size)] 

# Three particle events
startx = generate_startx(size=size)
starty = generate_startx(size=size)
theta = generate_theta_start_angle(size=size)
phi = generate_phi_angle(size=size)
zint = generate_zint(size=size)
theta1 = generate_theta_int_angle(size=size)
phi1 = generate_phi_angle(size=size)
theta2 = generate_theta_int_angle(size=size)
phi2 = generate_phi_angle(size=size)
theta3 = generate_theta_int_angle(size=size)
phi3 = generate_phi_angle(size=size)

test_3_params = [(startx[i], starty[i], theta[i], phi[i], zint[i], theta1[i], phi1[i], theta2[i], phi2[i], theta3[i], phi3[i]) for i in range(size)]
test_3_events = [_generate_three_event(*test_3_params[i]) for i in range(size)] 


# In[89]:


plot_3d(test_3_events[54])


# In[90]:


# запускать по желанию 
for i in range(0, 100, 10):
    plot_3d(test_3_events[i])


# In[16]:


for i in range(0, 100, 10):
    plot_3d(test_2_events[i])


# In[17]:


plot_3d(A)


# In[37]:


plot_3d(B)


# In[18]:


plot_3d(X)


# In[19]:


plot_3d(Y)


# ## Теперь нужно написать целевую функцию: близость модели к исходным данным

# In[22]:


# Будем минимизировать транспортную метрику (Вассерштайна)
# Она энергозатратная, но для разреженных матриц вроде бы не критично
def wasserstein_distance(mat1, mat2):
    # Get coordinates of 1s in each matrix
    coords1 = np.argwhere(mat1 == 1)  # shape (N, 3)
    coords2 = np.argwhere(mat2 == 1)  # shape (M, 3)
    
    if len(coords1) == 0 or len(coords2) == 0:
        distance = np.inf
    else:
        # Compute Euclidean cost matrix (distance between all pairs)
        cost_matrix = ot.dist(coords1, coords2, metric='euclidean')

        # Uniform weights (since all 1s are equally important)
        weights1 = np.ones(len(coords1)) / len(coords1)
        weights2 = np.ones(len(coords2)) / len(coords2)

        # Compute Wasserstein distance
        distance = ot.emd2(weights1, weights2, cost_matrix)
    return distance


# In[23]:


# Расстояние Хэмминга (симметрическая разность)
def hamming_distance(mat1, mat2):
    return np.sum(mat1 != mat2)


# In[24]:


# from skimage.metrics import structural_similarity as ssim

# # Структурная схожесть (Structural Similarity) -- вообще никак не работает
# def ssim(mat1, mat2):
#     return np.sum(mat1 != mat2)


# In[25]:


# Функция для сравнения с линией
def _objective_zero(params, to_x, to_y):
    startx, starty, theta, phi = params
    E = _generate_event(startx, starty, theta, phi, 21, 0, [], [])
    return wasserstein_distance(to_x, x_proj(E)) + wasserstein_distance(to_y, y_proj(E))
#     return ssim(EX, to_x, data_range=1.0) + ssim(EY, to_y, data_range=1.0)


# In[26]:


# Функция для сравнения со случаем, когда порождена одна частица
def _objective_one(params, to_x, to_y):
    startx, starty, theta, phi, zint, theta1, phi1 = params
    E = _generate_event(startx, starty, theta, phi, zint, 1, [theta1], [phi1])
    return wasserstein_distance(to_x, x_proj(E)) + wasserstein_distance(to_y, y_proj(E))

# def plot_3d_result_one(params):
#     startx, starty, theta, phi, zint, theta1, phi1 = params
#     X = _generate_event(startx, starty, theta, phi, zint, 1, [theta1], [phi1])
#     plot_3d(X)


# In[27]:


# Функция для сравнения со случаем, когда порождены две частицы
def _objective_two(params, to_x, to_y):
    startx, starty, theta, phi, zint, theta1, phi1, theta2, phi2 = params
    E = _generate_event(startx, starty, theta, phi, zint, 2, [theta1, theta2], [phi1, phi2])
    return wasserstein_distance(to_x, x_proj(E)) + wasserstein_distance(to_y, y_proj(E))

# def plot_3d_result_two(params):
#     startx, starty, theta, phi, zint, theta1, phi1, theta2, phi2 = params
#     X = _generate_event(startx, starty, theta, phi, zint, 2, [theta1, theta2], [phi1, phi2])
#     plot_3d(X)


# In[28]:


# Функция для сравнения со случаем, когда порождены три частицы
def _objective_three(params, to_x, to_y):
    startx, starty, theta, phi, zint, theta1, phi1, theta2, phi2, theta3, phi3 = params
    E = _generate_event(startx, starty, theta, phi, zint, 3, [theta1, theta2, theta3], [phi1, phi2, phi3])
    return wasserstein_distance(to_x, x_proj(E)) + wasserstein_distance(to_y, y_proj(E))

#     return hamming_distance(to_x, EX) + hamming_distance(to_y, EY)
#     return -ssim(EX, to_x, data_range=1.0) - ssim(EY, to_y, data_range=1.0)

# def plot_3d_result_three(params):
#     startx, starty, theta, phi, zint, theta1, phi1, theta2, phi2, theta3, phi3 = params
#     X = _generate_event(startx, starty, theta, phi, zint, 2, [theta1, theta2, theta3], [phi1, phi2, phi3])
#     plot_3d(X)


# In[105]:


# Функция для сравнения со случаем, когда порождены три частицы
def _objective_four(params, target_mask):
    if len(params) != 13:
        return 1e6
    try:
        gen_mask = _generate_four_event(*params) > 0
        return wasserstein_distance(gen_mask, target_mask)
    except Exception as e:
        print("Ошибка:", e)
        return 1e6



from scipy.spatial import cKDTree
def chamfer_loss(mask1: np.ndarray, mask2: np.ndarray) -> float:
    p1 = np.argwhere(mask1)
    p2 = np.argwhere(mask2)
    if len(p1) == 0 or len(p2) == 0:
        return np.inf
    d12 = cKDTree(p1).query(p2, k=1)[0].mean()
    d21 = cKDTree(p2).query(p1, k=1)[0].mean()
    return d12 + d21


def make_objective(gen_fun, target_mask):
    def f(params):
        p = list(params)
        # округляем координаты, где нужно
        p[0] = int(round(p[0]))   # x0
        p[1] = int(round(p[1]))   # y0
        p[4] = int(round(p[4]))   # zint
        try:
            gen_mask = gen_fun(*p) > 0
            return chamfer_loss(gen_mask, target_mask)
        except:
            return 1e6
    return f


def wemd(model_mask):
    P = np.argwhere(model_mask)       
    if len(P)==0 or len(coords_T)==0: return 1e6
    a = weight_T
    b = np.ones(len(P)) / len(P)
    M = ot.dist(coords_T, P)           
    return ot.emd2(a, b, M)           

def haus(m_bool):
    P = np.argwhere(m_bool)           
    if len(P) == 0 or len(coords_T)==0:
        return np.inf
    return max(
        directed_hausdorff(coords_T, P)[0],
        directed_hausdorff(P, coords_T)[0]
    )


def iou(model_mask):

    target_bool = np.zeros((96,96,44), bool)
    for x,y,z in coords_T.astype(int): target_bool[x,y,z]=1
    inter = np.logical_and(target_bool, model_mask).sum()
    union = np.logical_or(target_bool, model_mask).sum()
    return inter/union if union else 0.

def dice(model_mask):
    target_bool = np.zeros((96,96,44), bool)
    for x,y,z in coords_T.astype(int): target_bool[x,y,z]=1
    inter = np.logical_and(target_bool, model_mask).sum()
    return 2*inter / (target_bool.sum()+model_mask.sum()+1e-8)



def make_obj_energy(N):
    def f(p):
        p=list(p); p[0]=int(round(p[0])); p[1]=int(round(p[1])); p[4]=int(round(p[4]))
        try: mask = GEN[N](*p) > 0
        except: return 1e6
        return wemd(mask)
    return f

def proj(m, ttl):
    f,ax=plt.subplots(1,3,figsize=(10,3))
    ax[0].imshow(m.max(2)); ax[0].set_title('XY')
    ax[1].imshow(m.max(1)); ax[1].set_title('XZ')
    ax[2].imshow(m.max(0)); ax[2].set_title('YZ')
    f.suptitle(ttl); plt.tight_layout(); plt.show()


# In[106]:


def _generate_event(startx: float, starty: float,
                    theta0: float, phi0: float, zint: float,
                    npart: int,
                    theta_part: Sequence[float],
                    phi_part: Sequence[float]) -> np.ndarray:

    mask = np.zeros((96, 96, 44), dtype=np.uint8)

    def propagate(x0, y0, theta, phi, z_stop):
        z, x, y = 0, x0, y0
        while (
            z < z_stop
            and 0 <= z < 44
            and 0 <= x < 96
            and 0 <= y < 96
        ):
            mask[int(x), int(y), int(z)] = 1
            x += np.tan(theta) * np.cos(phi)
            y += np.tan(theta) * np.sin(phi)
            z += 1

    propagate(startx, starty, theta0, phi0, max(int(zint) - 1, 0))

    for th, ph in zip(theta_part, phi_part):
        propagate(startx, starty, th, ph, 44)

    return mask  


def _generate_zero_event(startx, starty, theta0, phi0, zint):
    return _generate_event(startx, starty, theta0, phi0, zint, 0, [], [])

def _generate_one_event(startx, starty, theta0, phi0, zint,
                        theta1, phi1):
    return _generate_event(startx, starty, theta0, phi0, zint,
                           1, [theta1], [phi1])

def _generate_two_event(startx, starty, theta0, phi0, zint,
                        theta1, phi1, theta2, phi2):
    return _generate_event(startx, starty, theta0, phi0, zint,
                           2, [theta1, theta2], [phi1, phi2])

def _generate_three_event(startx, starty, theta0, phi0, zint,
                          theta1, phi1, theta2, phi2, theta3, phi3):
    return _generate_event(startx, starty, theta0, phi0, zint,
                           3, [theta1, theta2, theta3],
                           [phi1,  phi2,  phi3])

def _generate_four_event(startx, starty, theta0, phi0, zint,
                         theta1, phi1, theta2, phi2, theta3, phi3,
                         theta4, phi4):
    return _generate_event(startx, starty, theta0, phi0, zint,
                           4, [theta1, theta2, theta3, theta4],
                           [phi1,  phi2,  phi3,  phi4])


def _generate_kink_event(startx, starty, theta0, phi0, zint,
                         k_break, npart,
                         theta1a, phi1a, theta1b, phi1b,
                         theta2a=None, phi2a=None, theta2b=None, phi2b=None,
                         theta3a=None, phi3a=None, theta3b=None, phi3b=None,
                         theta4a=None, phi4a=None, theta4b=None, phi4b=None):
    mask = np.zeros((96,96,44), np.uint8)
    def prop(x0,y0,th,ph,z0,z1):
        z,x,y = z0,x0,y0
        while z<z1 and 0<=x<96 and 0<=y<96 and z<44:
            mask[int(x),int(y),int(z)]=1
            x += np.tan(th)*np.cos(ph)
            y += np.tan(th)*np.sin(ph)
            z += 1
    prop(startx,starty,theta0,phi0,0,max(int(zint)-1,0))
    
    param_pairs = [(theta1a,phi1a,theta1b,phi1b),
                   (theta2a,phi2a,theta2b,phi2b),
                   (theta3a,phi3a,theta3b,phi3b),
                   (theta4a,phi4a,theta4b,phi4b)]
    for th_a,ph_a,th_b,ph_b in param_pairs[:npart]:
        prop(startx,starty,th_a,ph_a,0,int(k_break))
        prop(startx,starty,th_b,ph_b,int(k_break),44)
    return mask

def make_kink_gen(N):
    def g(*p):
        p=list(p)
        return _generate_kink_event(*(p[:5]+[p[5]]+[N]+p[6:]))
    return g



# In[ ]:


_bounds_xyz   = [(0, 95), (0, 95)]   
_bounds_theta_phi = [(0, np.pi), (-np.pi, np.pi)]  
_bounds_zint  = [(0, 44)]

bounds0 = _bounds_xyz + _bounds_theta_phi + _bounds_zint
bounds1 = _bounds_xyz + _bounds_theta_phi + _bounds_zint + \
          _bounds_theta_phi
bounds2 = _bounds_xyz + _bounds_theta_phi + _bounds_zint + \
          _bounds_theta_phi*2
bounds3 = _bounds_xyz + _bounds_theta_phi + _bounds_zint + \
          _bounds_theta_phi*3
bounds4 = _bounds_xyz + _bounds_theta_phi + _bounds_zint + \
          _bounds_theta_phi*4

target_mask = (_generate_four_event(
    45, 45, np.pi/6, 0, 18,
    np.pi/6, 0,
    np.pi/6, np.pi/3,
    np.pi/8, -np.pi/4,
    np.pi/4, np.pi/6
) > 0)


for name, gen_fun, bounds in [
    ("N=0", _generate_zero_event,  bounds0),
    ("N=1", _generate_one_event,   bounds1),
    ("N=2", _generate_two_event,   bounds2),
    ("N=3", _generate_three_event, bounds3),
    ("N=4", _generate_four_event,  bounds4),
]:
    obj = make_objective(gen_fun, target_mask)
    res = differential_evolution(obj, bounds, popsize=5, maxiter=20, disp=False)
    print(f"{name:4s}  |  loss = {res.fun:.3f}")


# In[31]:


obj = make_objective(_generate_one_event, target_mask)
result = differential_evolution(
    obj, bounds1,
    seed=42, maxiter=100, popsize=15, disp=True
)

print("восстан:", result.x)
print("потерь:", result.fun)

recovered_mask = _generate_one_event(*result.x) > 0

for z in [10, 20, 30]:
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].imshow(target_mask[:, :, z], cmap='gray')
    ax[0].set_title(f'Target z={z}')
    ax[1].imshow(recovered_mask[:, :, z], cmap='gray')
    ax[1].set_title(f'Recovered z={z}')
    plt.show()


# In[36]:


calo = np.load("calorimeter_response.npy", allow_pickle=True).item()


ids, counts = np.unique(calo["event_ID"], return_counts=True)
df_stats = pd.DataFrame({"event_ID": ids, "hit_count": counts})

star_df = df_stats[(df_stats.hit_count >= 50) & (df_stats.hit_count <= 200)]

print("События 50–200 hit:")
print(star_df)


# In[42]:


payload = np.load("calorimeter_response.npy", allow_pickle=True).item()
hits = pd.DataFrame({c: payload[c] for c in
                     ["event_ID","layer","index_along_x","index_along_y","energy_release"]})

EVENT_ID = 1.0
evt = hits[hits.event_ID == EVENT_ID]

coords_T, weight_T = [], []
for _,r in evt.iterrows():
    x,y,z = map(int,(r.index_along_x, r.index_along_y, r.layer))
    if 0<=x<96 and 0<=y<96 and 0<=z<44:
        coords_T.append((x,y,z))
        weight_T.append(r.energy_release)
coords_T = np.array(coords_T, dtype=float)
weight_T = np.array(weight_T, dtype=float)
weight_T /= weight_T.sum()         




best = {}
for N in (2,3,4):
    print(f"\n=== ENERGY-WASSER, N={N}")
    objE = make_obj_energy(N)
    de  = differential_evolution(objE, BOUNDS[N], popsize=25, maxiter=200, disp=False, seed=0)
    bh  = basinhopping(objE, de.x,
                       minimizer_kwargs=dict(method='L-BFGS-B', bounds=BOUNDS[N]),
                       niter=30, seed=0)
    best[N] = dict(params=bh.x, loss=bh.fun)

bestN = min(best, key=lambda n: best[n]["loss"])
maskR = GEN[bestN](*best[bestN]["params"]) > 0

print(f"\n>> BEST by energy-WEMD:  N={bestN},  loss={best[bestN]['loss']:.2f}")
print("IoU  :", iou(maskR))
print("Dice :", dice(maskR))
print("Haus :", haus(maskR))


proj_bool = np.zeros((96,96,44), bool)
for x,y,z in coords_T.astype(int): proj_bool[x,y,z]=1

proj(proj_bool, "TARGET  (bool)")
proj(maskR,     f"RECOVERED energyW (N={bestN})")


# In[104]:


def x0_random_kink(N):
    base=[np.random.randint(96),np.random.randint(96),
          np.random.rand()*np.pi, np.random.uniform(-np.pi,np.pi),
          np.random.randint(5,35), np.random.randint(10,40)]
    sec=[np.random.rand()*np.pi,
         np.random.uniform(-np.pi,np.pi)]*2*N
    return np.array(base+sec)

def x0_maxE_kink(N, df):
    sl0=df[df.layer==0]
    idx=sl0.energy_release.idxmax()
    x0,y0=df.loc[idx,["index_along_x","index_along_y"]]
    base=[int(x0),int(y0),
          np.pi/4,0,
          np.random.randint(5,35), np.random.randint(10,40)]
    sec=[np.random.rand()*np.pi,
         np.random.uniform(-np.pi,np.pi)]*2*N
    return np.array(base+sec)

def x0_hough_kink(N, df):
    # очень грубо: берём maxE точку как (x0,y0), theta=pi/4
    sl=df[df.layer<4][["index_along_x","index_along_y"]].values
    x0,y0=sl.mean(0)
    base=[x0,y0, np.pi/4, 0,
          np.random.randint(5,35), np.random.randint(10,40)]
    sec=[np.random.rand()*np.pi,
         np.random.uniform(-np.pi,np.pi)]*2*N
    return np.array(base+sec)


# In[93]:


get_ipython().system('pip install ace_tools')


# In[101]:


def start_hough_fixed(N, df):
    # собираем XY-точки первых 4 слоев
    pts = df[df.layer < 4][["index_along_x","index_along_y"]].astype(int).values
    # аккум для преобразования Хафа
    acc = np.zeros((96,96), dtype=int)
    for x,y in pts:
        if 0 <= x < 96 and 0 <= y < 96:
            acc[y,x] = 1
    # Hough-преобразование
    angles = np.linspace(-np.pi/2, np.pi/2, 360)
    h, th, d = hough_line(acc, angles)
    h_peaks, theta_peaks, d_peaks = hough_line_peaks(h, th, d)
    theta0 = float(theta_peaks[0]) if len(theta_peaks)>0 else np.pi/4
    phi0   = 0.0
    # стартовая точка как центр масс
    if pts.size:
        x0, y0 = pts.mean(axis=0)
    else:
        x0, y0 = 48, 48
    zint = int(df.energy_release.idxmax()) % 44
    base = [x0, y0, theta0, phi0, zint]
    # random-вторичные углы
    sec = []
    for _ in range(N):
        th_a = np.random.rand()*np.pi
        ph_a = np.random.uniform(-np.pi,np.pi)
        th_b = np.random.rand()*np.pi
        ph_b = np.random.uniform(-np.pi,np.pi)
        sec += [th_a, ph_a, th_b, ph_b]
    return np.array(base + sec)

# Выводим для N=4
N = 4
strategies = {
    "random": lambda N: start_random(N),
    "maxE"  : lambda N: start_maxE(N, evt),
    "hough" : lambda N: start_hough_fixed(N, evt),
}

for name, fn in strategies.items():
    x0 = fn(N)
    print(f"{name:>6s} start vector (len={len(x0)}):\n", x0, "\n")


# In[96]:


coords_R = np.argwhere(maskR)
fig = plt.figure(figsize=(7,7)); ax = fig.add_subplot(111,projection='3d')
ax.scatter(coords_T[:,0],coords_T[:,1],coords_T[:,2],
           c=weight_T, cmap='Blues', s=8, alpha=.6, label='data')
ax.scatter(coords_R[:,0],coords_R[:,1],coords_R[:,2],
           c='gold',s=12,marker='s',alpha=.9,label='reco')
ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z'); ax.invert_zaxis()
plt.legend(); plt.show()


# In[103]:


def load_mat_data(mat_path):
    if mat_path.endswith(".npy"):
        return np.load(mat_path, allow_pickle=True).item()
    elif mat_path.endswith(".mat"):
        raw = scipy.io.loadmat(mat_path)
        return {k: v.squeeze() for k, v in raw.items()
                if not k.startswith("__")}
    else:
        raise ValueError("Unsupported file format")

def build_volume(calo_data, event_id,
                 z_size=44, y_size=96, x_size=96):
    mask = calo_data['event_ID'] == event_id
    x = calo_data['index_along_x'][mask].astype(int)
    y = calo_data['index_along_y'][mask].astype(int)
    z = calo_data['layer'][mask].astype(int)
    e = calo_data['energy_release'][mask]

    vol = np.zeros((z_size, y_size, x_size), dtype=np.float32)
    for xx, yy, zz, ee in zip(x, y, z, e):
        if 0 <= xx < x_size and 0 <= yy < y_size and 0 <= zz < z_size:
            vol[zz, yy, xx] += ee
    coords_T = np.vstack([x, y, z]).T         
    weight_T = e.astype(float)
    return vol, coords_T, weight_T




file_path = "calorimeter_response.npy"
EVENT_ID = 1

calo_data = load_mat_data(file_path)



plot_event_with_reco(calo_data, EVENT_ID, maskR,
                     title_suffix="· blue=data  · yellow=fit")


# In[23]:


target_mask = (_generate_four_event(
    45, 45, np.pi/6, 0, 18,
    np.pi/6, 0,
    np.pi/6, np.pi/3,
    np.pi/8, -np.pi/4,
    np.pi/4, np.pi/6
) > 0)




result = differential_evolution(
    func=_objective_four,
    args=(target_mask,),
    bounds=bounds_four,     
    strategy='best1bin',
    maxiter=100,             
    popsize=25,              
    mutation=(0.4, 1.0),
    recombination=0.7,
    disp=True,
    seed=42,
    polish=True             
)
print("параметры:", result.x)
print("отклонение:", result.fun)


# In[24]:


def compare_masks(mask_true: np.ndarray, mask_pred: np.ndarray):
    print("Chamfer distance:", chamfer_distance(mask_true, mask_pred))
    print("Hausdorff distance:", hausdorff_distance(mask_true, mask_pred))
    print("IoU:", iou(mask_true, mask_pred))
    print("Dice:", dice(mask_true, mask_pred))



result = differential_evolution(
    func=_objective_four,
    args=(target_mask,),
    bounds=bounds_four,
    strategy='best1bin',
    maxiter=200,      
    popsize=40,      
    mutation=(0.4, 1.0),
    recombination=0.7,
    disp=True,
    seed=42,
    polish=True
)
params = list(result.x)
params[0] = int(round(params[0]))  
params[1] = int(round(params[1]))  
params[4] = int(round(params[4]))  
params = tuple(params)

recovered_mask = _generate_four_event(*params) > 0

metrics = compare_masks(target_mask, recovered_mask)
metrics


# In[ ]:


def make_objective(target_mask):
    def f(params):
        params = list(params)
        params[0] = int(round(params[0]))
        params[1] = int(round(params[1]))
        params[4] = int(round(params[4]))
        try:
            gen_mask = _generate_four_event(*params) > 0
            return wasserstein_distance(gen_mask, target_mask)
        except Exception as e:
            return 1e6
    return f

objective = make_objective(target_mask)

start = time.time()
result_de = differential_evolution(
    func=objective,
    bounds=bounds_four,
    strategy='best1bin',
    maxiter=100,
    popsize=30,
    seed=0,
    polish=True
)
time_de = time.time() - start

print("=== differential_evolution ===")
print("loss =", result_de.fun)
print("time =", time_de)

start = time.time()
x0 = result_de.x  
result_bh = basinhopping(
    objective,
    x0=x0,
    niter=30,
    stepsize=1.0,
    minimizer_kwargs={'method': 'L-BFGS-B', 'bounds': bounds_four},
    seed=0
)
time_bh = time.time() - start

print("\n=== basinhopping ===")
print("loss =", result_bh.fun)
print("time =", time_bh)

start = time.time()
result_shgo = shgo(
    func=objective,
    bounds=bounds_four,
    sampling_method='sobol'
)
time_shgo = time.time() - start

print("\n=== SHGO ===")
print("loss =", result_shgo.fun)
print("time =", time_shgo)


# In[28]:


result = opt.minimize(_objective_zero, x0=[1.0, 1.0, np.pi/4, np.pi/6], args=(x_proj(Z), y_proj(Z)),
                      bounds=[(0, 96), (0, 96), (0, np.pi/3), (-np.pi, np.pi)],
                      callback=lambda result: print(".", end=""),
                      method="Powell")
#                       method='COBYLA')
#                       method='Nelder-Mead')


# In[29]:


compare_XY(Z, _generate_zero_event(*result.x))


# In[30]:


result = opt.minimize(_objective_one, x0=[34.0, 52.0, 0, 0, 10, 0, 0], args=(Z.any,),
                      bounds=[(0, 96), (0, 96), (0, np.pi/3), (-np.pi, np.pi), (0, 21), (-np.pi / 2, np.pi / 2), (-np.pi, np.pi)],
                      callback=lambda result: print(".", end=""),
                      method='Nelder-Mead')


# In[31]:


result = opt.minimize(_objective_three, x0=[34.0, 45.0, 0, 0, 10, 0.0, 0.0, 0.1, 0.1, 0.2, 0.2], args=(X,),
                      bounds=[(0, 95), (0, 95), (0, np.pi/3), (-np.pi, np.pi), (0, 21), (-np.pi / 2, np.pi / 2), (-np.pi, np.pi), (-np.pi / 2, np.pi / 2), (-np.pi, np.pi), (-np.pi / 2, np.pi / 2), (-np.pi, np.pi)],
                      callback=lambda result: print(".", end=""),
                      method='Nelder-Mead', tol=0.000000001)


# In[50]:


result = opt.minimize(_objective_three, x0=[34.0, 45.0, 0, 0, 10, 0.0, 0.0, 0.2, 0.2, 0.5, 0.5], args=(x_proj(X), y_proj(X)),
                      bounds=[(0, 95), (0, 95), (0, np.pi/3), (-np.pi, np.pi), (0, 21), (-np.pi / 2, np.pi / 2), (-np.pi, np.pi), (-np.pi / 2, np.pi / 2), (-np.pi, np.pi), (-np.pi / 2, np.pi / 2), (-np.pi, np.pi)],
                      callback=lambda result: print(".", end=""),
                      method='Nelder-Mead',
                      options={
                         'maxiter': 10000,
                         'maxfev': 20000,
                         'xatol': 1e-6,
                         'fatol': 1e-6,
                         'disp': True,
                         'adaptive': True
                     })


# In[455]:


result = opt.minimize(_objective_three, x0=[34.0, 45.0, 0, 0, 10, 0.0, 0.0, 0.2, 0.2, -0.5, -0.5], args=(x_proj(X), y_proj(X)),
                      bounds=[(0, 95), (0, 95), (0, np.pi/3), (-np.pi, np.pi), (0, 21), (-np.pi / 2, np.pi / 2), (-np.pi, np.pi), (-np.pi / 2, np.pi / 2), (-np.pi, np.pi), (-np.pi / 2, np.pi / 2), (-np.pi, np.pi)],
                      callback=lambda result: print(".", end=""),
                      method='Powell',
                      options={
                         'maxiter': 10000,
                         'maxfev': 20000,
                         'xtol': 1e-6,
                         'ftol': 1e-6,
                         'disp': True
                     })


# In[43]:


result = opt.minimize(_objective_three, x0=[34.0, 45.0, 0, 0, 10, 0.0, 0.0, 0.2, 0.2, 0.5, 0.5], args=(x_proj(X), y_proj(X)),
                      bounds=[(0, 95), (0, 95), (0, np.pi/3), (-np.pi, np.pi), (0, 21), (-np.pi / 2, np.pi / 2), (-np.pi, np.pi), (-np.pi / 2, np.pi / 2), (-np.pi, np.pi), (-np.pi / 2, np.pi / 2), (-np.pi, np.pi)],
                      callback=lambda result: print(".", end=""),
                      method='cobyla',
                      options={
                         'maxiter': 100000,
                         'rhobeg': 0.2,
                         'tol': 1e-6,
                         'disp': True
                     })
                      # method='Powell')


# In[24]:


# Этот метод работает практически идеально, но долга

X = test_3_events[70]

def diff_callback(xk, convergence):
    current_min = _objective_three(xk, x_proj(X), y_proj(X))
    print(r"f(x) = {0:.4f}".format(current_min))
    return current_min < 0.1  # Stop if condition met

result = opt.differential_evolution(_objective_three, args=(x_proj(X), y_proj(X)),
                      bounds=[(0, 95), (0, 95), (0, np.pi/3), (-np.pi, np.pi), (0, 21), (-np.pi / 2, np.pi / 2), (-np.pi, np.pi), (-np.pi / 2, np.pi / 2), (-np.pi, np.pi), (-np.pi / 2, np.pi / 2), (-np.pi, np.pi)],
                      callback=diff_callback)
                      # method='Powell')


# In[25]:


result


# In[65]:


result.x


# In[26]:


compare_XY(X, _generate_three_event(*result.x))


# In[66]:


compare_proj(X, _generate_three_event(*result.x))

