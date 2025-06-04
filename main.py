from less import * 
 
num_data = 20000 
num_bins = 100 
 
set_distr('G') 
set_num_distr(False) 
 
 
 
 
# параметры для отрицательных 
 
w_neg = [2.925e-05, 0.158] 
params_neg = [1.00, 1.00, 1.00, 
          2.956, 0.905, 4.063, 
          0.965, 0.870, 1.430] 
 
w_neg = [2.76792725e-01, 6.82648789e-01] 
params_neg = [2.36822133e+00, 8.70148938e-01, 3.64500811e+00, 
          8.74311812e-01, 8.68558512e-01, 1.57729958e+00, 
          3.89726376e+00, 8.31168852e-01, 4.21781715e+03] 
 
 
 
 
 
# параметры для общей части 
w_gen = [0.502, 0.276] 
params_gen = [0.999, 0.710, 1.271, 
              0.951, 0.489, 3.321, 
              0.545, 2.651, 1.076 
              ] 
 
# параметры для позитивных чисел 
w_pos = [0.2924917,  0.10092523] 
params_pos = [0.59317036, 0.1, 1., 
              3.31154415, 1.4174199, 6.15965185, 
              1.55782399, 0.88593041, 1.] 
 
#model1 = MixtureG(w_neg, params_neg) # негатив: ГГГ 
#model2 = MixtureW(w_pos, params_pos) # позитив: ВГВ 
#model = MixMixture(0.5, model1, model2) 
 
model = MixtureW(w_gen, params_gen) 
data = read_data() 
#data = data[data > 0] 
 
plt.figure(2) 
pdf = model.pdf(data)                                                                                        
# здесь 
cdf = model.cdf(data)                                                                                        
# здесь 
 
plt.hist(data, bins=num_bins, density=True, alpha=0.5, label='Данные') 
plt.plot(data, pdf, label="Смесь из смесей распределения", linewidth=2) 
plt.xlabel('Колебания курса') 
plt.ylabel('Плотность') 
plt.title('Гистограмма колебаний биткоина') 
plt.legend() 
plt.grid() 

plt.figure(3) 
cdf.sort() 
ax1 = sns.ecdfplot(data, label='Эксперимент.') 
ax1.axes.set_yticks(np.arange(start=0, stop=1.1, step=0.1)) 
ax1.grid() 
plt.plot(data, cdf, label='Теоретическая') 
plt.title(f"Кумулятивные функции\nСмесь из смесей распределения") 
plt.xlabel("Колебания курса") 
plt.ylabel("Вероятность") 
plt.legend() 
 
 
#plt.show() 
 
observed_counts, expected_counts = calculate_observed_and_expected(data, 
num_bins, model) 
print("Критерий хи-квадрат: p_value = ", chisquare(observed_counts, 
expected_counts)[1]) 
print("Критерий Колмогорова-Смирнова: p-value = ", ks_2samp(observed_counts, 
expected_counts)[1]) 
max_delta = max(abs(observed_counts - expected_counts)) 
delta = [] 
x = np.linspace(min(data), max(data), len(data)) 
 
np.random.seed(10) 
for i in range(num_data): 
    gen_data = np.array(sorted(np.random.choice(x, size=len(data), 
p=model.pdf(x)/model.pdf(x).sum()))) 
    gen_observed_counts, gen_expected_counts = 
calculate_observed_and_expected(gen_data, num_bins, model) 
    delta.append(max(abs(gen_observed_counts - gen_expected_counts))) 
 
delta = np.array(delta) 
freq = len(delta[delta > max_delta]) 
print("Частота случаев, когда разница между теор. и экспер. для искусственных 
данных больше разницы для исходных данных: ", freq) 
print("Всего количество сгенерированных датасетов: ", len(delta)) 
print("Макс. разница исходной выборки: ", max_delta) 
 
print("Критерий хи-квадрат: p_value = ", chisquare(observed_counts, 
expected_counts)[1]) 
print("Критерий Колмогорова-Смирнова: p-value = ", ks_2samp(observed_counts, 
expected_counts)[1]) 
 
 
 
plt.figure(4) 
plt.hist(delta, bins=30) 
#plt.scatter(range(num_data), delta) 
plt.plot(np.full(shape=num_data//2, fill_value=max_delta, dtype=np.float64), 
range(num_data//2), color='red', linestyle='dashed', label=f'max delta = 
{round(max_delta, 4)}') 
plt.title(f"График максимальных разниц между теор. и эксп. значениями 
частот\nдля искусственных данных") 
plt.legend() 
plt.grid() 
plt.show() 