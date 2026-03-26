import matplotlib.pyplot as plt
import numpy as np

# Данные из эксперимента
approaches = ['Rule-based', 'XGBoost + SHAP']
accuracy = [81.2, 96.4]
frr = [18.7, 3.8]          # False Rejection Rate (ложные отказы)

x = np.arange(len(approaches))
width = 0.35

fig, ax1 = plt.subplots(figsize=(10, 6))

# Столбцы для Accuracy
bars1 = ax1.bar(x - width/2, accuracy, width, label='Accuracy (%)', color='#2E86C1')
ax1.set_ylabel('Accuracy (%)', color='#2E86C1')
ax1.set_ylim(75, 100)

# Вторая ось Y для FRR
ax2 = ax1.twinx()
bars2 = ax2.bar(x + width/2, frr, width, label='Ложные отказы (FRR, %)', color='#E74C3C')
ax2.set_ylabel('Ложные отказы FRR (%)', color='#E74C3C')
ax2.set_ylim(0, 25)

# Настройки
plt.title('Сравнительный анализ Accuracy и уровня ложных отказов (FRR)', fontsize=14, pad=20)
ax1.set_xticks(x)
ax1.set_xticklabels(approaches, fontsize=12)

# Добавляем значения на столбцы
for bar in bars1:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5, f'{height:.1f}%', 
             ha='center', va='bottom', fontsize=11)

for bar in bars2:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5, f'{height:.1f}%', 
             ha='center', va='bottom', fontsize=11)

# Легенда
fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0.92), ncol=2)

plt.grid(axis='y', alpha=0.3)
plt.tight_layout()

# Сохраняем изображение 
plt.savefig('figures/comparison_accuracy_frr.png', dpi=300, bbox_inches='tight')
plt.show()

print("Диаграмма сохранена как 'figures/comparison_accuracy_frr.png'")