from math import exp, log, gamma 
import random 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from scipy.optimize import minimize, Bounds, LinearConstraint 
from scipy.stats import chisquare, ks_2samp 
from scipy import stats 
from scipy.special import gammainc 
import seaborn as sns 
 
num_bins = 50 
is_two_distrib = False 
distr = 'G' 
labl = "" 
 
def set_distr(d): 
    global distr 
    distr = d 
 
def set_num_distr(yes_no): 
    global is_two_distrib 
    is_two_distrib = yes_no 
 
def scaler(data): 
    mean = np.mean(data) 
    std = np.std(data) 
    data_sc = [(data[i] - mean) / std for i in range(len(data))] 
    return data_sc 
 
def read_data(): 
    data = pd.read_csv('dataBTC.csv', encoding='utf-8', usecols=['timestamp', 
'high']) 
    data['datetime'] = pd.to_datetime(data['timestamp']) 
    data = data.drop(columns=['timestamp']) 
 
    data['datetime'] = pd.to_datetime(data['datetime']) 
    data = data.sort_values(by='datetime') 
    data['time_diff'] = data['datetime'].diff().dt.total_seconds() / 60 
    data = data[data['time_diff'] == 1.0] 
    data['course_diff'] = data['high'].diff() 
    q_25 = data['course_diff'].quantile(0.15) 
    q_75 = data['course_diff'].quantile(0.85) 
    dq = q_75 - q_25 
    vibr_ind = data['course_diff'][ 
        ((data['course_diff'] < q_25 - 1.7 * dq) | (data['course_diff'] > 
q_75 + 1.7 * dq))].index 
    data.loc[vibr_ind, 'course_diff'] = None 
    data = data.dropna(ignore_index=True) 
    data = data.sort_values(by='course_diff') 
    data = sorted(scaler(data['course_diff'])) 
    window = 10 
    data = np.convolve(data, np.ones(window) / window, 'valid') 
    return np.array(data) 
 
class MixMixture: 
    def __init__(self, w, model1, model2): 
        self.model1 = model1 
        self.model2 = model2 
 
        self.w = w
 def pdf(self, x): 
        return self.w * self.model1.pdf(x) + (1 - self.w) * 
self.model2.pdf(x)                                             # здесь 
 
    def cdf(self, x): 
        x = np.abs(x) 
        return self.w * self.model1.cdf(x) + (1 - self.w) * 
self.model2.cdf(x) 
 
class Weibull: 
    def __init__(self, teta, p, c): 
        self.teta = teta 
        self.p = p 
 
    def pdf(self, x): 
        coefficient = self.teta / self.p 
        result = np.where(x >= 0, 
                          coefficient * (x / self.p) ** (self.teta - 1) * 
np.exp(-(x / self.p) ** self.teta), 
                          0) 
        return result 
 
    def cdf(self, x): 
        return np.where(x >= 0, 
                        1 - np.exp(-(x / self.p) ** self.teta), 
                        0) 
 
 
class GeneralizedGamma: 
    def __init__(self, teta, p, c): 
        self.teta = teta 
        self.p = p 
        self.c = c 
 
    def pdf(self, x): 
        coefficient = (self.c / (self.teta ** self.p)) / gamma(self.p / 
self.c) 
        result = np.where(x >= 0, 
                          coefficient * x ** (self.p - 1) * np.exp(-(x / 
self.teta) ** self.c), 
                          0) 
        return result 
 
    def cdf(self, x): 
        x = np.abs(x) 
        return np.where(x >= 0, 
                        gammainc(self.p / self.c, (x / self.teta) ** self.c), 
                        0) 
 
class MixtureG: 
    def __init__(self, w, params): 
 
 
        self.one = GeneralizedGamma(params[0], params[1], params[2]) 
        self.two = GeneralizedGamma(params[3], params[4], params[5]) 
        if not is_two_distrib: 
            self.three = GeneralizedGamma(params[6], params[7], params[8]) 
 
        self.w = w 
 
    if is_two_distrib: 
        def pdf(self, x): 
            return self.w * self.one.pdf(-x) + (1 - self.w) * self.two.pdf(x)                                                       
# здесь
 def cdf(self, x): 
            return self.w * self.one.cdf(x) + (1 - self.w) * self.two.cdf(x) 
    else: 
        def pdf(self, x): 
 
            return self.w[0] * self.one.pdf(-x) + self.w[1] * self.two.pdf(
x) + (1 - self.w[0] - self.w[1]) * self.three.pdf(-x)     # здесь 
 
        def cdf(self, x): 
            return self.w[0] * self.one.cdf(x) + self.w[1] * self.two.cdf(x) 
+ (1 - self.w[0] - self.w[1]) * self.three.cdf(x) 
 
class MixtureW: 
    def __init__(self, w, params): 
 
 
        self.one = Weibull(params[0], params[1], params[2]) 
        self.two = GeneralizedGamma(params[3], params[4], params[5]) 
        if not is_two_distrib: 
            self.three = GeneralizedGamma(params[6], params[7], params[8]) 
 
 
        self.w = w 
 
    if is_two_distrib: 
        def pdf(self, x): 
            return self.w * self.one.pdf(-x) + (1 - self.w) * self.two.pdf(x)                                                       
# здесь 
 
        def cdf(self, x): 
            return self.w * self.one.cdf(x) + (1 - self.w) * self.two.cdf(x) 
    else: 
        def pdf(self, x): 
 
            return self.w[0] * self.one.pdf(-x) + self.w[1] * self.two.pdf(x) 
+ (1 - self.w[0] - self.w[1]) * self.three.pdf(x)     # здесь 
 
        def cdf(self, x): 
            return self.w[0] * self.one.cdf(x) + self.w[1] * self.two.cdf(x) 
+ (1 - self.w[0] - self.w[1]) * self.three.cdf(x) 
 
 
def negative_log_likelihood_three(params, x): 
    if is_two_distrib: 
        if not 0 < params[0] < 1: 
            return 1e10 
        if params[1] <= 0 or params[2] <= 0 or params[3] <= 0 or params[4] <= 
0 or params[5] <= 0 or params[6] <= 0: 
            return 1e10 
        mix = Mixture(params[0], params[1:]) 
    else: 
        if not (0 < params[0] < 1 and 0 < params[1] < 1 and (params[0] + 
params[1]) < 1): 
            return 1e10 
        if params[2] <= 0 or params[3] <= 0 or params[4] <= 0 or \ 
           params[5] <= 0 or params[6] <= 0 or params[7] <= 0 or \ 
           params[8] <= 0 or params[9] <= 0 or params[10] <= 0: 
            return 1e10 
        mix = Mixture(params[0:2], params[2:]) 
    pdf = mix.pdf(x) 
    return -np.sum(np.log(pdf)) 
 
def ks1_asympt(cdf_data):
    n=len(cdf_data) 
    cdf_data.sort() 
    D_n=0 
    for i in range(n): 
        if max(abs(cdf_data[i]-i/n),abs(cdf_data[i]-(i-1)/n))>D_n: 
            D_n=max(abs(cdf_data[i]-i/n),abs(cdf_data[i]-(i-1)/n)) 
    return stats.kstwobign.sf(n**0.5*D_n) 
 
def calculate_observed_and_expected(data, num_bins, distribution): 
    observed_counts, bin_edges = np.histogram(data, bins=num_bins, 
density=False) 
 
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:]) 
    expected_counts = np.zeros(num_bins) 
 
    for i in range(num_bins): 
        expected_probs = distribution.pdf(bin_centers[i])  # Ожидаемая 
плотность 
        expected_counts[i] = expected_probs * len(data) * (bin_edges[1] - 
bin_edges[0]) 
 
    expected_counts /= expected_counts.sum() 
 
    return np.array(observed_counts), np.array(expected_counts) * len(data) 
 
 
 
if is_two_distrib: 
    initial_params = [0.5, 
                      1.0, 1.0, 1.0, 
                      1.0, 1.0, 1.0] 
 
    bounds_three = Bounds( 
        [0.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], 
        [1.0, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf] 
    ) 
 
    linear_constraint_three = LinearConstraint( 
        [1, 0, 0, 0, 0, 0, 0], 
        0.0, 0.9999 
    ) 
 
else: 
    initial_params = [0.5, 0.5, 
                      1.0, 1.0, 1.0, 
                      1.0, 1.0, 1.0, 
                      1.0, 1.0, 1.0] 
 
    bounds_three = Bounds( 
        [0.0, 0.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], 
        [1.0, 1.0, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 
np.inf, np.inf] 
    ) 
 
    linear_constraint_three = LinearConstraint( 
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
        0.0, 0.9999 
    ) 
 
data = read_data() 
data = data[data > 0] 
 
result = minimize( 
    negative_log_likelihood_three, 
    initial_params, 
    args=(data,),                                                                                               
# здесь 
    method='SLSQP', 
    bounds=bounds_three, 
    constraints=[linear_constraint_three], 
    options={'maxiter': 1000, 'ftol': 1e-9, 'disp': True} 
) 
 
plt.figure(1) 
if is_two_distrib: 
    best_mix = Mixture(result.x[0], result.x[1:]) 
else: 
    best_mix = Mixture(result.x[0:2], result.x[2:]) 
pdf = best_mix.pdf(data)                                                                                        
# здесь 
cdf = best_mix.cdf(data)                                                                                        
# здесь 
 
plt.hist(data, bins=num_bins, density=True, alpha=0.5, label='Данные') 
if labl == "": 
    if is_two_distrib and distr == 'W': 
        labl = "Смесь 2 компонент: одно об. гамма-распределение и одно 
Вейбулла" 
    elif is_two_distrib and distr == 'G': 
        labl = "Смесь 2 компонент: два об. гамма-распределения" 
    elif not is_two_distrib and distr == 'W': 
        labl = "Смесь 3-х компонент: два об. гамма-распределения и одно 
распр. Вейбулла" 
    elif not is_two_distrib and distr == 'G': 
        labl = "Смесь 3-х компонент: три об. гамма-распределения" 
 
plt.plot(data, pdf, label=labl, linewidth=2) 
plt.xlabel('Колебания курса') 
plt.ylabel('Плотность') 
plt.title('Гистограмма колебаний биткоина') 
plt.legend() 
plt.grid() 
 
plt.figure(2) 
cdf.sort() 
ax1 = sns.ecdfplot(data, label='Эксперимент.') 
ax1.axes.set_yticks(np.arange(start=0, stop=1.1, step=0.1)) 
ax1.grid() 
plt.plot(data, cdf, label='Теоретическая') 
plt.title(f"Кумулятивные функции\n{labl}") 
plt.xlabel("Колебания курса") 
plt.ylabel("Вероятность") 
plt.legend() 
 
print("Параметры: ", result.x) 
observed, expected = calculate_observed_and_expected(data, num_bins, 
best_mix)                                  # здесь 
 
print("Критерий хи-квадрат: p-value = ", chisquare(observed, expected)[1]) 
print("Критерий Колмогорова-Смирнова: p-value = ", ks_2samp(observed, 
expected)[1]) 
 
plt.show()