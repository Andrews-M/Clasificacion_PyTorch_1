#!/usr/bin/env python
# coding: utf-8

# In[54]:


from sklearn.datasets import make_circles
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn
import numpy as np


# # Generando datos

# In[55]:


# Vamos a generar 100 muestras siguiendo un patron circular
n = 1000 # muestras

X, y = make_circles(n,
                    noise=0.03,
                    random_state=42) 

print(f"Primeros 10 elementos de X:\n{X[:10]}")
print(f"\n Primeros 10 elementos de y:\n{y[:10]}")


# In[56]:


# Generando un DataFrame de los datos

datos = pd.DataFrame({"X1": X[:, 0],
    "X2": X[:, 1],
    "label": y
})
datos.head(10)


# In[57]:


# Contando los datos
datos.label.value_counts()


# In[58]:


# Visualizando RESALTAR LA NORMALIZACION DE LOS DATOS
plt.scatter(x=X[:, 0], 
            y=X[:, 1], 
            c=y, 
            cmap=plt.cm.RdYlBu);


# In[59]:


# Forma de los datos
X.shape, y.shape


# In[60]:


X_sample = X[0]
y_sample = y[0]
print(f"Valores para una muestra de X: {X_sample}; para y: {y_sample}")
print(f"Forma de una muestra de X: {X_sample.shape}; para y: {y_sample.shape}")


# # Convirtiendo a tensores

# In[61]:


#Los datos estan con tipo NumPy array; queremos pasarlos a tensores de PyTorch

X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)

X[:5], y[:5]


# # Separando los datos

# In[62]:


# Separando en "train data" y "test data" 80 - 20
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=0.2, 
                                                    random_state=42) 

len(X_train), len(X_test), len(y_train), len(y_test)


# # Definiendo el modelo

# In[63]:


class Modelo_Circulo(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer_1 = nn.Linear(in_features=2, out_features=10) # toma 2 elementos (entradas) y produce 5
        self.layer_2 = nn.Linear(in_features=10, out_features=10) # toma 5 elementos y produce 1 (la salida "y")
        self.layer_3 = nn.Linear(in_features=10, out_features=1)
    
    def forward(self, x): # Método forward.El cálculo se desplaza de la capa 1 a la capa 2.
        
        return self.layer_3(self.layer_2(self.layer_1(x))) # los datos pasan primero por la capa 1 y luego van hacia la capa 2

modelo = Modelo_Circulo()
modelo


# In[64]:


# Función de pérdidas
# nn.BCELoss no tiene la capa sigmoid incluida
# https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html

loss_fn = nn.BCEWithLogitsLoss() # Tiene la capa sigmoid en la misma clase

# Optimizador
optimizer = torch.optim.SGD(params=modelo.parameters(), 
                            lr=0.1)


# In[65]:


# Métrica de evaluación de clasificación: Precisión (ACCURACY)
def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item() # torch.eq() calcula igualdad en los tensores; se suman y luego .item() regresa el dato como valor estándar
    acc = (correct / len(y_pred)) * 100 
    return acc


# # Predicciones sin entrenar

# In[66]:


# Primeras predicciones
# logits: Los datos "crudos" (raw data) que salen del modelo
with torch.inference_mode():
    
    y1_logits = modelo(X_test)
    
print(f"tamaño de las predicciones: {len(y1_logits)}, Forma: {y1_logits.shape}")
print(f"tamaño de las muestras test: {len(y_test)}, Forma: {y_test.shape}")
print(f"\nPrimeras diez predicciones:\n{y1_logits[:10]}")
print(f"\nPrimeras etiquetas:\n{y_test[:10]}")


# In[67]:


# Usando sigmoid en datos "logits". Esto ayudará a interpretarlos mejor, ya que hay que convertir los logits a datos similares a las etiquetas
y_pred_probs = torch.sigmoid(y1_logits)
y_pred_probs[:10] # datos en forma de probabilidades


# In[68]:


# Si dato >= 0.5 entonces y = 1; Si dato < 0.5, entonces y = 0
y_preds = torch.round(y_pred_probs)

# Eliminamos la dimension extra
y_preds = y_preds.squeeze()
y_preds[:10]


# In[69]:


y_preds.shape


# In[70]:


y_test[:10]


# In[71]:


accuracy_fn(y_test, y_preds)


# # Entrenando el modelo

# In[72]:


torch.manual_seed(42)

epochs = 1000

# Crear listas de pérdidas vacías para realizar un seguimiento de los valores
train_loss_values = []
test_loss_values = []
epoch_count = []
train_acc_count = []
test_acc_count = []

# Bucles de entrenamiento y evaluacion
for epoch in range(epochs):
    
    # ENTRENAMIENTO
    
    modelo.train()

    # 1. Forward pass
    y_logits = modelo(X_train).squeeze() 
    y_pred = torch.round(torch.sigmoid(y_logits)) 
  
    # 2. Calcular Pérdidas (Loss) y Presición (Accuracy)
 
    loss = loss_fn(y_logits, 
                   y_train) 
    
    acc = accuracy_fn(y_true=y_train, 
                      y_pred=y_pred) 

    # 3. Optimizer
    optimizer.zero_grad()

    # 4. Loss backwards
    loss.backward()

    # 5. Optimizer step
    optimizer.step()

    #EVALUACION
    
    modelo.eval()
    with torch.inference_mode():
        # 1. Forward pass
        test_logits = modelo(X_test).squeeze() 
        test_pred = torch.round(torch.sigmoid(test_logits))
        # 2. Calculo de loss y accuracy
        test_loss = loss_fn(test_logits,
                            y_test)
        test_acc = accuracy_fn(y_true=y_test,
                               y_pred=test_pred)

    # Desplegando cada 10 epocas
    if epoch % 10 == 0:
        epoch_count.append(epoch)
        train_loss_values.append(loss.detach().numpy())
        test_loss_values.append(test_loss.detach().numpy())
        train_acc_count.append(acc)
        test_acc_count.append(test_acc)
        print(f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%")


# In[73]:


#Mostrar resultados del entrenamiento - PERDIDAS
plt.figure(figsize=(10, 7))
plt.xticks(fontsize = 25 )
plt.yticks(fontsize = 25)

plt.plot(epoch_count, train_loss_values,'b', label="Train loss")
plt.plot(epoch_count, test_loss_values,'r', label="Test loss")

plt.ylabel('Losses', size = 34)
plt.xlabel('Epoch', size = 34)
plt.legend(prop={"size": 18});

plt.grid()
plt.show()


# In[74]:


train_acc_count =np.array(train_acc_count).T
#Mostrar resultados del entrenamiento - Accuracy
plt.figure(figsize=(10, 7))
plt.xticks(fontsize = 25 )
plt.yticks(fontsize = 25)

plt.plot(epoch_count, train_acc_count,'b', label="Train acc")
plt.plot(epoch_count, test_acc_count,'r', label="Test acc")

plt.ylabel('Accuracy', size = 34)
plt.xlabel('Epoch', size = 34)
plt.legend(prop={"size": 18});

plt.grid()
plt.show()


# In[75]:


# Predicciones
modelo.eval()
with torch.inference_mode():
    y_prediccion = torch.round(torch.sigmoid(modelo(X_test))).squeeze()
y_prediccion[:10], y[:10] # y_prediccion y etiquetas en el mismo formato


# In[76]:


X_test_numpy_tensor = X_test.numpy()
y_pred_numpy_tensor = y_prediccion.numpy() 

# Visualizando RESALTAR LA NORMALIZACION DE LOS DATOS
plt.scatter(x=X_test_numpy_tensor[:, 0], 
            y=X_test_numpy_tensor[:, 1], 
            c=y_pred_numpy_tensor, 
            cmap=plt.cm.RdYlBu);


# # Segundo modelo

# In[77]:


class Modelo_Circulo_2(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer_1 = nn.Linear(in_features=2, out_features=10) # toma 2 elementos (entradas) y produce 5
        self.layer_2 = nn.Linear(in_features=10, out_features=10) # toma 5 elementos y produce 1 (la salida "y")
        self.layer_3 = nn.Linear(in_features=10, out_features=1)
        self.relu = nn.ReLU() # <- se incluye la funcion de activacion ReLU 
        
    
    def forward(self, x):
        
        return self.layer_3(self.relu(self.layer_2(self.relu(self.layer_1(x))))) # Se intercala la funcion ReLu entre las capas de la red

modelo2 = Modelo_Circulo_2()
modelo2


# In[78]:


# Función de pérdidas
loss_fn = nn.BCEWithLogitsLoss() # Tiene la capa sigmoid en la misma clase

# Optimizador
optimizer = torch.optim.SGD(params=modelo2.parameters(), 
                            lr=0.1)


# In[79]:


torch.manual_seed(42)

epochs = 1200

# Crear listas de pérdidas vacías para realizar un seguimiento de los valores
train_loss_values = []
test_loss_values = []
epoch_count = []
train_acc_count = []
test_acc_count = []

# Bucles de entrenamiento y evaluacion
for epoch in range(epochs):
    
    # ENTRENAMIENTO
    
    modelo2.train()

    # 1. Forward pass
    y_logits = modelo2(X_train).squeeze() 
    y_pred = torch.round(torch.sigmoid(y_logits)) 
  
    # 2. Calcular Pérdidas (Loss) y Presición (Accuracy)
 
    loss = loss_fn(y_logits, 
                   y_train) 
    
    acc = accuracy_fn(y_true=y_train, 
                      y_pred=y_pred) 

    # 3. Optimizer
    optimizer.zero_grad()

    # 4. Loss backwards
    loss.backward()

    # 5. Optimizer step
    optimizer.step()

    #EVALUACION
    
    modelo2.eval()
    with torch.inference_mode():
        # 1. Forward pass
        test_logits = modelo2(X_test).squeeze() 
        test_pred = torch.round(torch.sigmoid(test_logits))
        # 2. Calculo de loss y accuracy
        test_loss = loss_fn(test_logits,
                            y_test)
        test_acc = accuracy_fn(y_true=y_test,
                               y_pred=test_pred)

    # Desplegando cada 10 epocas
    if epoch % 10 == 0:
        epoch_count.append(epoch)
        train_loss_values.append(loss.detach().numpy())
        test_loss_values.append(test_loss.detach().numpy())
        train_acc_count.append(acc)
        test_acc_count.append(test_acc)
        print(f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%")


# In[80]:


#Mostrar resultados del entrenamiento - PERDIDAS
plt.figure(figsize=(10, 7))
plt.xticks(fontsize = 25 )
plt.yticks(fontsize = 25)

plt.plot(epoch_count, train_loss_values,'b', label="Train loss")
plt.plot(epoch_count, test_loss_values,'r', label="Test loss")

plt.ylabel('Losses', size = 34)
plt.xlabel('Epoch', size = 34)
plt.legend(prop={"size": 18});

plt.grid()
plt.show()


# In[81]:


train_acc_count =np.array(train_acc_count).T
#Mostrar resultados del entrenamiento - Accuracy
plt.figure(figsize=(10, 7))
plt.xticks(fontsize = 25 )
plt.yticks(fontsize = 25)

plt.plot(epoch_count, train_acc_count,'b', label="Train acc")
plt.plot(epoch_count, test_acc_count,'r', label="Test acc")

plt.ylabel('Accuracy', size = 34)
plt.xlabel('Epoch', size = 34)
plt.legend(prop={"size": 18});

plt.grid()
plt.show()


# In[82]:


# Predicciones
modelo2.eval()
with torch.inference_mode():
    y_prediccion = torch.round(torch.sigmoid(modelo2(X_test))).squeeze()
y_prediccion[:10], y[:10] # y_prediccion y etiquetas en el mismo formato


# In[83]:


X_test_numpy_tensor = X_test.numpy()
y_pred_numpy_tensor = y_prediccion.numpy() 

# Visualizando RESALTAR LA NORMALIZACION DE LOS DATOS
plt.scatter(x=X_test_numpy_tensor[:, 0], 
            y=X_test_numpy_tensor[:, 1], 
            c=y_pred_numpy_tensor, 
            cmap=plt.cm.RdYlBu);


# In[ ]:




